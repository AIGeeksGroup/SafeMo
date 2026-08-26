#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from closd.diffusion_planner.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler
from closd.utils import hf_handler

from .eval_data import (
    DataConfig,
    build_catalog,
    build_loader,
    evaluator_collate,
    file_sha256,
    ids_sha256,
    read_ids,
)
from .eval_metrics import evaluate_once
from .modeling import (
    FixedLoraNegation,
    checkpoint_lora_spec,
    inject_named_lora,
    load_main_table_checkpoint,
    main_table_args,
)
from .reporting import build_summary, summary_text


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 32
CONTEXT_LEN = 20
PRED_LEN = 40
GUIDANCE_ARGUMENT = 7.5


class GeneratedDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]], source_dataset):
        self.records = records
        self.source_dataset = source_dataset
        self.w_vectorizer = source_dataset.w_vectorizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        motion = self.source_dataset.inv_transform(record["motion"])
        motion = (
            motion - self.source_dataset.mean_for_eval
        ) / self.source_dataset.std_for_eval
        words, poses = [], []
        for token in record["tokens"]:
            word, pose = self.w_vectorizer[token]
            words.append(word[None, :])
            poses.append(pose[None, :])
        return (
            np.concatenate(words, axis=0),
            np.concatenate(poses, axis=0),
            record["caption"],
            record["cap_len"],
            motion,
            record["length"],
            "_".join(record["tokens"]),
        )


def first_existing(explicit: Optional[str], candidates: Iterable[Path], label: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("{} not found: {}".format(label, path))
        return path
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not locate {}. Tried:\n  {}".format(label, "\n  ".join(map(str, candidates)))
    )


def move_condition(condition, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in condition.items()
    }


def generate_loader(model, diffusion, source_loader, workers: int) -> DataLoader:
    args = SimpleNamespace(
        autoregressive=True,
        autoregressive_include_prefix=False,
        pred_len=PRED_LEN,
        context_len=CONTEXT_LEN,
    )
    sample_fn = AutoRegressiveSampler(args, diffusion.p_sample_loop).sample
    records: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for motion, kwargs in tqdm(source_loader, desc="sampling", leave=False):
            kwargs["y"] = move_condition(kwargs["y"], dist_util.dev())
            motion = motion.to(dist_util.dev())
            kwargs["y"]["scale"] = torch.full(
                (motion.shape[0],), GUIDANCE_ARGUMENT, device=dist_util.dev()
            )
            sample = sample_fn(
                model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            kwargs["y"]["lengths"][:] = sample.shape[-1]
            for index in range(motion.shape[0]):
                tokens = kwargs["y"]["tokens"][index].split("_")
                records.append(
                    {
                        "motion": sample[index].squeeze().permute(1, 0).cpu().numpy(),
                        "length": int(kwargs["y"]["lengths"][index].item()),
                        "caption": kwargs["y"]["text"][index],
                        "tokens": tokens,
                        "cap_len": tokens.index("eos/OTHER") + 1,
                    }
                )
    dataset = GeneratedDataset(records, source_loader.dataset)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=evaluator_collate,
        drop_last=True,
        num_workers=workers,
    )


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate SafeMo MMU on HumanML3D")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--dependencies-root", required=True)
    result.add_argument("--humanml-root", required=True)
    result.add_argument("--unsafe-ids-file", required=True)
    result.add_argument(
        "--bert-model",
        default="distilbert/distilbert-base-uncased",
        help="Local DistilBERT directory or Hugging Face model ID",
    )
    result.add_argument("--dataset-scope", choices=["test", "all"], required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--loader-workers", type=int, default=8)
    result.add_argument("--generated-workers", type=int, default=4)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--static-alpha", type=float, default=1.0)
    result.add_argument("--gated-forget-alpha", type=float, default=2.0)
    result.add_argument("--gated-retain-alpha", type=float, default=0.05)
    result.add_argument("--model-mean")
    result.add_argument("--model-std")
    result.add_argument("--eval-mean")
    result.add_argument("--eval-std")
    result.add_argument("--use-raw-model", action="store_true")
    result.add_argument("--allow-test-list-on-all", action="store_true")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    cli = parser().parse_args()
    if cli.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if cli.loader_workers < 0 or cli.generated_workers < 0:
        raise ValueError("worker counts must be >= 0")
    alphas = [cli.static_alpha, cli.gated_forget_alpha, cli.gated_retain_alpha]
    if not np.isfinite(alphas).all() or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be finite and non-negative")

    checkpoint = Path(cli.checkpoint).expanduser().resolve()
    dependencies = Path(cli.dependencies_root).expanduser().resolve()
    humanml_root = Path(cli.humanml_root).expanduser().resolve()
    unsafe_file = Path(cli.unsafe_ids_file).expanduser().resolve()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    for label, path in (
        ("checkpoint", checkpoint),
        ("dependencies root", dependencies),
        ("HumanML3D root", humanml_root),
        ("unsafe IDs", unsafe_file),
    ):
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(label, path))
    if output_dir.exists() and not cli.resume:
        raise FileExistsError(
            "Output already exists; choose a new path or pass --resume: {}".format(output_dir)
        )

    if cli.dataset_scope == "all" and not cli.allow_test_list_on_all:
        unsafe_ids = set(read_ids(unsafe_file)[0])
        test_ids = set(read_ids(humanml_root / "test.txt")[0])
        if unsafe_ids and unsafe_ids <= test_ids:
            raise ValueError(
                "Unsafe list appears test-only; use --dataset-scope test or explicitly "
                "pass --allow-test-list-on-all"
            )

    model_mean = first_existing(
        cli.model_mean,
        [dependencies / "data/dataset/HumanML3D/Mean.npy", humanml_root / "Mean.npy"],
        "model Mean.npy",
    )
    model_std = first_existing(
        cli.model_std,
        [dependencies / "data/dataset/HumanML3D/Std.npy", humanml_root / "Std.npy"],
        "model Std.npy",
    )
    eval_mean = first_existing(
        cli.eval_mean,
        [
            REPO_ROOT / "closd/diffusion_planner/dataset/t2m_mean.npy",
            dependencies / "closd/diffusion_planner/dataset/t2m_mean.npy",
            dependencies / "evaluators/t2m/Comp_v6_KLD005/meta/mean.npy",
        ],
        "evaluator Mean.npy",
    )
    eval_std = first_existing(
        cli.eval_std,
        [
            REPO_ROOT / "closd/diffusion_planner/dataset/t2m_std.npy",
            dependencies / "closd/diffusion_planner/dataset/t2m_std.npy",
            dependencies / "evaluators/t2m/Comp_v6_KLD005/meta/std.npy",
        ],
        "evaluator Std.npy",
    )
    config = DataConfig(
        humanml_root=humanml_root,
        unsafe_ids_file=unsafe_file,
        scope=cli.dataset_scope,
        glove_dir=dependencies / "glove",
        model_mean=model_mean,
        model_std=model_std,
        eval_mean=eval_mean,
        eval_std=eval_std,
    )
    for path in (
        config.glove_dir / "our_vab_data.npy",
        config.glove_dir / "our_vab_words.pkl",
        config.glove_dir / "our_vab_idx.pkl",
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing GloVe dependency: {}".format(path))
    evaluator_checkpoint = dependencies / "evaluators/t2m/text_mot_match/model/finest.tar"
    if not evaluator_checkpoint.is_file():
        raise FileNotFoundError("Missing evaluator checkpoint: {}".format(evaluator_checkpoint))

    catalog, data_diagnostics = build_catalog(config)
    fixseed(cli.seed)
    hf_handler.get_dependencies = lambda: str(dependencies)
    dist_util.setup_dist(cli.device)
    from closd.diffusion_planner.model import mdm as mdm_module
    from closd.diffusion_planner.model.BERT.BERT_encoder import load_bert as real_load_bert

    bert_candidate = Path(cli.bert_model).expanduser()
    bert_model = str(bert_candidate.resolve()) if bert_candidate.exists() else cli.bert_model
    mdm_module.load_bert = lambda _legacy_path: real_load_bert(bert_model)
    bootstrap = build_loader(
        config, catalog, "forget", "eval", BATCH_SIZE, cli.loader_workers
    )
    for subset in ("forget", "retain"):
        source = build_loader(config, catalog, subset, "eval", BATCH_SIZE, 0)
        ground_truth = build_loader(config, catalog, subset, "gt", BATCH_SIZE, 0)
        if len(source.dataset) // BATCH_SIZE < 1 or len(ground_truth.dataset) // BATCH_SIZE < 1:
            raise RuntimeError("{} has fewer than 32 effective samples".format(subset))

    model, diffusion = create_model_and_diffusion(main_table_args(), bootstrap)
    lora_spec = checkpoint_lora_spec(checkpoint, not cli.use_raw_model)
    replaced = inject_named_lora(
        model,
        lora_spec["modules"],
        rank=lora_spec["rank"],
        alpha=lora_spec["alpha"],
        dropout=0.0,
    )
    checkpoint_info = load_main_table_checkpoint(model, checkpoint, not cli.use_raw_model)
    checkpoint_info["replaced_layers"] = replaced
    model.to(dist_util.dev()).eval()
    eval_wrapper = EvaluatorMDMWrapper("humanml", dist_util.dev())

    identity = {
        "format_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_component": checkpoint_info["component"],
        "unsafe_ids_sha256": file_sha256(unsafe_file),
        "unsafe_ids_resolved_sha256": ids_sha256(read_ids(unsafe_file)[0]),
        "dataset_scope": cli.dataset_scope,
        "scope_ids_sha256": data_diagnostics["scope_ids_sha256"],
        "catalog_ids_sha256": data_diagnostics["catalog_ids_sha256"],
        "seed": cli.seed,
        "replication_seeds": [cli.seed + index for index in range(cli.repetitions)],
        "repetitions": cli.repetitions,
        "loader_workers": cli.loader_workers,
        "generated_workers": cli.generated_workers,
        "bert_model": bert_model,
        "batch_size": BATCH_SIZE,
        "context_len": CONTEXT_LEN,
        "pred_len": PRED_LEN,
        "output_frames": 196,
        "sampling_protocol": "legacy-table2",
        "requested_guidance_scale": GUIDANCE_ARGUMENT,
        "effective_cfg_scale": 1.0,
        "static_alpha": cli.static_alpha,
        "gated_forget_alpha": cli.gated_forget_alpha,
        "gated_retain_alpha": cli.gated_retain_alpha,
        "model_mean_sha256": file_sha256(model_mean),
        "model_std_sha256": file_sha256(model_std),
        "eval_mean_sha256": file_sha256(eval_mean),
        "eval_std_sha256": file_sha256(eval_std),
        "evaluator_checkpoint_sha256": file_sha256(evaluator_checkpoint),
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("evaluate.py", "eval_data.py", "eval_metrics.py", "modeling.py")
        },
    }
    identity_hash = json_sha256(identity)
    protocol = {
        "identity": identity,
        "identity_sha256": identity_hash,
        "sources": {
            "checkpoint": str(checkpoint),
            "humanml_root": str(humanml_root),
            "unsafe_ids_file": str(unsafe_file),
            "dependencies_root": str(dependencies),
            "model_mean": str(model_mean),
            "model_std": str(model_std),
            "eval_mean": str(eval_mean),
            "eval_std": str(eval_std),
            "evaluator_checkpoint": str(evaluator_checkpoint),
        },
        "checkpoint_load": checkpoint_info,
        "data_diagnostics": data_diagnostics,
        "gating": "precomputed prompt-level routing labels",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity_hash:
            raise RuntimeError("Cannot resume: protocol identity differs")
    elif cli.resume and any(output_dir.iterdir()):
        raise RuntimeError("Cannot resume non-empty output without protocol.json")
    atomic_json(protocol_path, protocol)
    if cli.preflight_only:
        print(json.dumps(protocol, indent=2, ensure_ascii=False))
        return

    cells = [
        ("static_forget", "SafeMo-Static", "forget", float(cli.static_alpha)),
        ("static_retain", "SafeMo-Static", "retain", float(cli.static_alpha)),
        ("gated_forget", "SafeMo-Gated", "forget", float(cli.gated_forget_alpha)),
        ("gated_retain", "SafeMo-Gated", "retain", float(cli.gated_retain_alpha)),
    ]
    raw_path = output_dir / "raw_metrics.json"
    raw = (
        json.loads(raw_path.read_text(encoding="utf-8"))
        if raw_path.exists()
        else {
            "format_version": 1,
            "protocol_identity_sha256": identity_hash,
            "repetitions": cli.repetitions,
            "cells": {},
        }
    )
    if raw.get("protocol_identity_sha256") != identity_hash:
        raise RuntimeError("Cannot resume: raw metrics use another protocol")

    for cell_key, method, subset, alpha in cells:
        completed = {int(item["replication"]) for item in raw["cells"].get(cell_key, [])}
        for replication in range(cli.repetitions):
            if replication in completed:
                print("[resume] {} replication {}".format(cell_key, replication))
                continue
            run_seed = cli.seed + replication
            fixseed(run_seed)
            source = build_loader(
                config,
                catalog,
                subset,
                "eval",
                BATCH_SIZE,
                cli.loader_workers,
            )
            ground_truth = build_loader(
                config,
                catalog,
                subset,
                "gt",
                BATCH_SIZE,
                cli.loader_workers,
            )
            eligible = len(source.dataset)
            effective = (eligible // BATCH_SIZE) * BATCH_SIZE
            print(
                "[eval] {} subset={} alpha={:g} repetition={} seed={} samples={}/{}".format(
                    method, subset, alpha, replication, run_seed, effective, eligible
                )
            )
            generated_name = "{}_a{:g}".format(cell_key, alpha)
            generated = generate_loader(
                FixedLoraNegation(model, alpha),
                diffusion,
                source,
                cli.generated_workers,
            )
            metrics = evaluate_once(
                eval_wrapper, ground_truth, generated, generated_name
            )
            raw["cells"].setdefault(cell_key, []).append(
                {
                    "replication": replication,
                    "seed": run_seed,
                    "method": method,
                    "subset": subset,
                    "alpha": alpha,
                    "eligible_samples": eligible,
                    "effective_samples": effective,
                    "ground_truth_eligible_samples": len(ground_truth.dataset),
                    "ground_truth_effective_samples": (
                        len(ground_truth.dataset) // BATCH_SIZE
                    ) * BATCH_SIZE,
                    "metrics": metrics,
                }
            )
            raw["cells"][cell_key].sort(key=lambda item: int(item["replication"]))
            atomic_json(raw_path, raw)

    summary = build_summary(raw, cells, identity_hash)
    atomic_json(output_dir / "summary.json", summary)
    rendered = summary_text(summary, protocol)
    (output_dir / "summary.txt").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
