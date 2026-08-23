from __future__ import annotations

import codecs
import hashlib
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from closd.diffusion_planner.data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from closd.diffusion_planner.data_loaders.tensors import collate as base_collate


_VECTORIZERS: Dict[str, WordVectorizer] = {}


def get_vectorizer(path: Path) -> WordVectorizer:
    key = str(path.resolve())
    if key not in _VECTORIZERS:
        _VECTORIZERS[key] = WordVectorizer(key, "our_vab")
    return _VECTORIZERS[key]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join("{}\n".format(value) for value in ids).encode("utf-8")
    ).hexdigest()


def normalize_id(value: str) -> str:
    value = value.lstrip("\ufeff").strip()
    if not value:
        return ""
    value = value.split()[0].split(",")[0]
    name = Path(value).name
    return Path(name).stem if Path(name).suffix.lower() in {".npy", ".txt"} else name


def read_ids(path: Path) -> Tuple[List[str], int]:
    values = [normalize_id(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    seen: Set[str] = set()
    ordered: List[str] = []
    duplicates = 0
    for value in values:
        if value in seen:
            duplicates += 1
        else:
            seen.add(value)
            ordered.append(value)
    return ordered, duplicates


@dataclass(frozen=True)
class DataConfig:
    humanml_root: Path
    unsafe_ids_file: Path
    scope: str
    glove_dir: Path
    model_mean: Path
    model_std: Path
    eval_mean: Path
    eval_std: Path
    unit_length: int = 4
    max_text_len: int = 20
    max_motion_length: int = 196


def build_catalog(config: DataConfig) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    scope_file = config.humanml_root / "{}.txt".format(config.scope)
    scope_ids, scope_duplicates = read_ids(scope_file)
    unsafe_ids_ordered, unsafe_duplicates = read_ids(config.unsafe_ids_file)
    unsafe_ids = set(unsafe_ids_ordered)
    scope_set = set(scope_ids)

    if config.scope == "all":
        split_sets = []
        for split in ("train", "val", "test"):
            path = config.humanml_root / "{}.txt".format(split)
            if not path.is_file():
                raise FileNotFoundError("Missing canonical split file: {}".format(path))
            split_sets.append(set(read_ids(path)[0]))
        union = set().union(*split_sets)
        if union != scope_set:
            raise RuntimeError("all.txt is not the union of train.txt, val.txt, and test.txt")

    text_dir = config.humanml_root / "texts"
    motion_dir = config.humanml_root / "new_joint_vecs"
    records: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {
        "scope": config.scope,
        "scope_file": str(scope_file),
        "scope_ids": len(scope_ids),
        "scope_duplicates": scope_duplicates,
        "unsafe_list_ids": len(unsafe_ids_ordered),
        "unsafe_duplicates": unsafe_duplicates,
        "unsafe_in_scope": len(scope_set & unsafe_ids),
        "unsafe_outside_scope": len(unsafe_ids - scope_set),
        "missing_text": 0,
        "missing_motion": 0,
        "bad_motion": 0,
    }
    for sample_id in scope_ids:
        text_path = text_dir / "{}.txt".format(sample_id)
        motion_path = motion_dir / "{}.npy".format(sample_id)
        if not text_path.is_file():
            diagnostics["missing_text"] += 1
            continue
        if not motion_path.is_file():
            diagnostics["missing_motion"] += 1
            continue
        try:
            shape = np.load(str(motion_path), mmap_mode="r").shape
            if len(shape) != 2 or shape[1] != 263:
                raise ValueError(shape)
            length = int(shape[0])
        except Exception:
            diagnostics["bad_motion"] += 1
            continue
        records.append(
            {
                "id": sample_id,
                "text_path": text_path,
                "motion_path": motion_path,
                "length": length,
                "is_unsafe": sample_id in unsafe_ids,
            }
        )
    diagnostics["catalog_size"] = len(records)
    diagnostics["catalog_unsafe"] = sum(bool(record["is_unsafe"]) for record in records)
    diagnostics["catalog_retain"] = len(records) - int(diagnostics["catalog_unsafe"])
    diagnostics["scope_ids_sha256"] = ids_sha256(scope_ids)
    diagnostics["catalog_ids_sha256"] = ids_sha256([str(record["id"]) for record in records])
    if not records:
        raise RuntimeError("No valid HumanML3D records in selected scope")
    if not diagnostics["unsafe_in_scope"]:
        raise RuntimeError("Unsafe-ID file has zero overlap with selected scope")
    return records, diagnostics


class HumanMLEvalDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        catalog: Sequence[Dict[str, object]],
        mode: str,
        subset: str,
        fixed_len: int = 0,
    ) -> None:
        if mode not in {"gt", "eval"} or subset not in {"forget", "retain"}:
            raise ValueError("Invalid mode/subset: {}/{}".format(mode, subset))
        self.config = config
        self.mode = mode
        self.subset = subset
        self.dataset_name = "t2m"
        self.dataname = "t2m"
        self.num_actions = 1
        self.opt = SimpleNamespace(
            fixed_len=int(fixed_len),
            max_motion_length=int(fixed_len or config.max_motion_length),
            max_text_len=config.max_text_len,
            unit_length=config.unit_length,
            joints_num=22,
            dim_pose=263,
            disable_offset_aug=(mode == "eval" and fixed_len > 0),
        )
        if mode == "gt":
            self.mean = np.load(str(config.eval_mean))
            self.std = np.load(str(config.eval_std))
        else:
            self.mean = np.load(str(config.model_mean))
            self.std = np.load(str(config.model_std))
            self.mean_for_eval = np.load(str(config.eval_mean))
            self.std_for_eval = np.load(str(config.eval_std))
        statistic_names = ["mean", "std"]
        if mode == "eval":
            statistic_names += ["mean_for_eval", "std_for_eval"]
        for name in statistic_names:
            value = getattr(self, name)
            if value.shape != (263,) or not np.isfinite(value).all():
                raise ValueError("{} must be finite with shape (263,)".format(name))
        if np.any(self.std == 0) or (
            mode == "eval" and np.any(self.std_for_eval == 0)
        ):
            raise ValueError("Normalization std contains zero")

        minimum = fixed_len if fixed_len > 0 else 20
        wanted = subset == "forget"
        self.entries = [
            record for record in catalog
            if bool(record["is_unsafe"]) == wanted and int(record["length"]) >= minimum
        ]
        self.entries.sort(key=lambda record: int(record["length"]))
        self.effective_ids = [str(record["id"]) for record in self.entries]
        self.w_vectorizer = get_vectorizer(config.glove_dir)
        self.t2m_dataset = self
        self.mean_gpu = torch.tensor(self.mean)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std)[None, :, None, None]
        if not self.entries:
            raise RuntimeError("No {} records of length >= {}".format(subset, minimum))

    def inv_transform(self, value):
        if torch.is_tensor(value):
            mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
            std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
            return value * std + mean
        return value * self.std + self.mean

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _caption(path: Path, max_text_len: int) -> Tuple[str, List[str], int]:
        with codecs.open(str(path), "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        parts = (random.choice(lines) if lines else "").split("#")
        caption = (parts[0] if parts else "").strip() or "empty"
        if len(parts) > 1 and parts[1].strip():
            base_tokens = parts[1].strip().split(" ")
        else:
            base_tokens = [
                "{}/OTHER".format(word.replace("/", "-")) for word in caption.split()
            ]
        base_tokens = base_tokens[:max_text_len]
        tokens = ["sos/OTHER"] + base_tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens += ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        return caption, tokens, sent_len

    def __getitem__(self, index: int):
        record = self.entries[index]
        caption, tokens, sent_len = self._caption(
            Path(record["text_path"]), self.opt.max_text_len
        )
        word_embeddings, pos_one_hots = [], []
        for token in tokens:
            word_embedding, pos_one_hot = self.w_vectorizer[token]
            word_embeddings.append(word_embedding[None, :])
            pos_one_hots.append(pos_one_hot[None, :])

        full_length = int(record["length"])
        if self.opt.fixed_len > 0:
            motion_length = self.opt.fixed_len
        else:
            double = np.random.choice([False, False, True])
            units = full_length // self.opt.unit_length - int(double)
            motion_length = max(
                self.opt.unit_length,
                min(units * self.opt.unit_length, self.opt.max_motion_length),
            )
        available = max(0, full_length - motion_length)
        start = random.randint(0, available)
        if self.opt.disable_offset_aug:
            start = random.randint(0, min(self.opt.unit_length, available))
        motion = np.load(str(record["motion_path"]))[start : start + motion_length]
        if motion.shape != (motion_length, 263):
            raise RuntimeError("Invalid crop for {}".format(record["id"]))
        motion = (motion - self.mean) / self.std
        if motion_length < self.opt.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.opt.max_motion_length - motion_length, 263))],
                axis=0,
            )
        return (
            np.concatenate(word_embeddings, axis=0),
            np.concatenate(pos_one_hots, axis=0),
            caption,
            sent_len,
            motion,
            motion_length,
            "_".join(tokens),
            str(record["id"]),
            motion,
            bool(record["is_unsafe"]),
        )


def prefix_collate(batch: Sequence[tuple], pred_len: int):
    adapted, originals, flags = [], [], []
    for item in batch:
        tensor = torch.tensor(item[8].T).float().unsqueeze(1)
        suffix = tensor[..., -pred_len:]
        adapted.append(
            {
                "inp": suffix,
                "prefix": tensor[..., :-pred_len],
                "text": item[2],
                "tokens": item[6],
                "lengths": pred_len,
                "key": item[7],
            }
        )
        originals.append(suffix)
        flags.append(bool(item[9]))
    motion, condition = base_collate(adapted)
    condition["y"]["orig"] = torch.stack(originals, dim=0)
    condition["y"]["refined"] = condition["y"]["orig"]
    condition["y"]["is_unsafe"] = torch.tensor(flags).bool()
    return motion, condition


def evaluator_collate(batch: List[tuple]):
    batch.sort(key=lambda item: item[3], reverse=True)
    return default_collate(batch)


def build_loader(
    config: DataConfig,
    catalog: Sequence[Dict[str, object]],
    subset: str,
    mode: str,
    batch_size: int,
    workers: int,
    context_len: int = 20,
    pred_len: int = 40,
) -> DataLoader:
    fixed_len = context_len + pred_len if mode == "eval" else 0
    dataset = HumanMLEvalDataset(config, catalog, mode, subset, fixed_len)
    collate_fn = partial(prefix_collate, pred_len=pred_len) if mode == "eval" else evaluator_collate
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
