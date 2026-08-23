from __future__ import annotations

import codecs
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from closd.diffusion_planner.data_loaders.humanml.utils.word_vectorizer import WordVectorizer

from .eval_data import ids_sha256, read_ids


@dataclass(frozen=True)
class TrainDataConfig:
    humanml_root: Path
    unsafe_ids_file: Path
    glove_dir: Path
    mean: Path
    std: Path
    split: str = "train"
    context_len: int = 20
    pred_len: int = 40
    crop_policy: str = "legacy-prefix"
    max_text_len: int = 20
    unit_length: int = 4


def build_train_catalog(config: TrainDataConfig) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    split_file = config.humanml_root / "{}.txt".format(config.split)
    ids, split_duplicates = read_ids(split_file)
    unsafe_ordered, unsafe_duplicates = read_ids(config.unsafe_ids_file)
    unsafe = set(unsafe_ordered)
    minimum = config.context_len + config.pred_len
    records = []
    missing_text = missing_motion = invalid_motion = 0
    for sample_id in tqdm(ids, desc="indexing HumanML3D"):
        text = config.humanml_root / "texts" / "{}.txt".format(sample_id)
        motion = config.humanml_root / "new_joint_vecs" / "{}.npy".format(sample_id)
        if not text.is_file():
            missing_text += 1
            continue
        if not motion.is_file():
            missing_motion += 1
            continue
        try:
            shape = np.load(str(motion), mmap_mode="r").shape
            if len(shape) != 2 or shape[1] != 263 or shape[0] < minimum:
                raise ValueError(shape)
        except Exception:
            invalid_motion += 1
            continue
        records.append(
            {
                "id": sample_id,
                "text": text,
                "motion": motion,
                "length": int(shape[0]),
                "unsafe": sample_id in unsafe,
            }
        )
    diagnostics = {
        "split": config.split,
        "split_ids": len(ids),
        "split_duplicates": split_duplicates,
        "unsafe_ids": len(unsafe_ordered),
        "unsafe_duplicates": unsafe_duplicates,
        "unsafe_in_split": len(set(ids) & unsafe),
        "unsafe_outside_split": len(unsafe - set(ids)),
        "valid_records": len(records),
        "valid_unsafe": sum(bool(record["unsafe"]) for record in records),
        "valid_safe": sum(not bool(record["unsafe"]) for record in records),
        "missing_text": missing_text,
        "missing_motion": missing_motion,
        "invalid_or_short_motion": invalid_motion,
        "split_ids_sha256": ids_sha256(ids),
        "catalog_ids_sha256": ids_sha256([str(record["id"]) for record in records]),
    }
    if not records or not diagnostics["valid_unsafe"] or not diagnostics["valid_safe"]:
        raise RuntimeError("Training split must contain valid unsafe and safe samples")
    return records, diagnostics


class HumanMLTrainDataset(Dataset):
    def __init__(
        self,
        config: TrainDataConfig,
        records: Sequence[Dict[str, object]],
        unsafe: bool,
    ) -> None:
        self.config = config
        self.records = [record for record in records if bool(record["unsafe"]) == unsafe]
        self.mean = np.load(str(config.mean))
        self.std = np.load(str(config.std))
        if self.mean.shape != (263,) or self.std.shape != (263,) or np.any(self.std == 0):
            raise ValueError("HumanML3D Mean.npy and Std.npy must have shape (263,)")
        self.vectorizer = WordVectorizer(str(config.glove_dir), "our_vab")
        if not self.records:
            raise RuntimeError("Selected training stream is empty")

    def __len__(self) -> int:
        return len(self.records)

    def _text(self, path: Path):
        with codecs.open(str(path), "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        parts = (random.choice(lines) if lines else "").split("#")
        caption = (parts[0] if parts else "").strip() or "empty"
        if len(parts) > 1 and parts[1].strip():
            base = parts[1].strip().split()
        else:
            base = ["{}/OTHER".format(word.replace("/", "-")) for word in caption.split()]
        base = base[: self.config.max_text_len]
        tokens = ["sos/OTHER"] + base + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens += ["unk/OTHER"] * (self.config.max_text_len + 2 - sent_len)
        words, poses = [], []
        for token in tokens:
            word, pose = self.vectorizer[token]
            words.append(word[None, :])
            poses.append(pose[None, :])
        return (
            caption,
            "_".join(tokens),
            sent_len,
            np.concatenate(words, axis=0),
            np.concatenate(poses, axis=0),
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        total = self.config.context_len + self.config.pred_len
        available = int(record["length"]) - total
        if self.config.crop_policy == "legacy-prefix":
            start = random.randint(0, min(self.config.unit_length, available))
        elif self.config.crop_policy == "random":
            start = random.randint(0, available)
        else:
            raise ValueError("Unknown crop policy: {}".format(self.config.crop_policy))
        motion = np.load(str(record["motion"]))[start : start + total]
        motion = ((motion - self.mean) / self.std).astype(np.float32, copy=False)
        caption, tokens, sent_len, words, poses = self._text(Path(record["text"]))
        return {
            "id": str(record["id"]),
            "motion": motion,
            "caption": caption,
            "tokens": tokens,
            "sent_len": sent_len,
            "word_embs": words.astype(np.float32, copy=False),
            "pos_ohot": poses.astype(np.float32, copy=False),
        }


def train_collate(batch: Sequence[Dict[str, object]], context_len: int, pred_len: int):
    full = torch.stack(
        [torch.from_numpy(item["motion"].T).unsqueeze(1) for item in batch], dim=0
    )
    suffix = full[..., context_len : context_len + pred_len]
    lengths = torch.full((len(batch),), pred_len, dtype=torch.long)
    condition = {
        "y": {
            "mask": torch.ones((len(batch), 1, 1, pred_len), dtype=torch.bool),
            "lengths": lengths,
            "prefix": full[..., :context_len],
            "text": [str(item["caption"]) for item in batch],
            "tokens": [str(item["tokens"]) for item in batch],
            "db_key": [str(item["id"]) for item in batch],
            "word_embs": torch.stack(
                [torch.from_numpy(item["word_embs"]) for item in batch], dim=0
            ),
            "pos_ohot": torch.stack(
                [torch.from_numpy(item["pos_ohot"]) for item in batch], dim=0
            ),
            "sent_lens": torch.tensor(
                [int(item["sent_len"]) for item in batch], dtype=torch.long
            ),
        }
    }
    return suffix, condition


def build_train_loader(
    config: TrainDataConfig,
    records: Sequence[Dict[str, object]],
    unsafe: bool,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    dataset = HumanMLTrainDataset(config, records, unsafe)
    if len(dataset) < batch_size:
        raise RuntimeError("Training stream has fewer samples than batch size")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        generator=generator,
        collate_fn=partial(
            train_collate, context_len=config.context_len, pred_len=config.pred_len
        ),
    )
