from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from closd.diffusion_planner.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from closd.utils import hf_handler

from .eval_data import file_sha256
from .modeling import (
    checkpoint_lora_spec,
    freeze_for_lora,
    inject_lora,
    load_backbone_checkpoint,
    main_table_args,
    set_lora_scale,
)
from .train_data import TrainDataConfig, build_train_catalog, build_train_loader
from .train_losses import decouple_motion, kinematic_losses, masked_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, str(temporary))
    temporary.replace(path)


def first_file(explicit: Optional[str], candidates: Iterable[Path], label: str) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError("{} not found: {}".format(label, candidate))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate {}. Tried:\n  {}".format(label, "\n  ".join(map(str, candidates)))
    )


def clone_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    return value


def move_optimizer(optimizer: torch.optim.Optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def endless(loader):
    while True:
        for batch in loader:
            yield batch


def without_text_encoder(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if not key.startswith("clip_model.")}


def load_state(model: nn.Module, state: Dict[str, torch.Tensor], label: str) -> None:
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("clip_model.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "{} mismatch: missing={}, unexpected={}".format(
                label, missing[:20], list(result.unexpected_keys)[:20]
            )
        )


class PreHeadCapture:
    def __init__(self, model: nn.Module):
        self.value = None
        self.handle = model.output_process.poseFinal.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        self.value = inputs[0]

    def pop_bdt(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("Pre-head feature hook did not run")
        value = self.value
        self.value = None
        return value.permute(1, 2, 0).unsqueeze(2)


class TextMotionAligner:
    def __init__(self, device):
        self.wrapper = EvaluatorMDMWrapper("humanml", device)
        for module in (
            self.wrapper.text_encoder,
            self.wrapper.motion_encoder,
            self.wrapper.movement_encoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.wrapper.text_encoder.eval()
        self.wrapper.motion_encoder.train()
        self.wrapper.movement_encoder.train()
        self.device = device

    def loss(
        self,
        words: torch.Tensor,
        poses: torch.Tensor,
        sentence_lengths: torch.Tensor,
        motions: torch.Tensor,
        motion_lengths: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            order = torch.argsort(sentence_lengths, descending=True)
            text_sorted = self.wrapper.text_encoder(
                words.index_select(0, order).float(),
                poses.index_select(0, order).float(),
                sentence_lengths.index_select(0, order).long(),
            )
            text = text_sorted.index_select(0, torch.argsort(order))
        order = torch.argsort(motion_lengths, descending=True)
        sequence = motions.index_select(0, order).squeeze(2).permute(0, 2, 1)
        movement = self.wrapper.movement_encoder(sequence[..., :-4])
        motion = self.wrapper.motion_encoder(
            movement, motion_lengths.index_select(0, order).long() // 4
        )
        text = F.normalize(text.index_select(0, order), dim=-1)
        motion = F.normalize(motion, dim=-1)
        labels = torch.arange(text.shape[0], device=self.device)
        return 0.5 * (
            F.cross_entropy(text @ motion.t() / temperature, labels)
            + F.cross_entropy(motion @ text.t() / temperature, labels)
        )


class Trainer:
    def __init__(
        self,
        args,
        model,
        model_avg,
        baseline,
        diffusion,
        optimizer,
        unsafe_loader,
        safe_loader,
        aligner,
        output_dir: Path,
        start_step: int,
        lora_modules: Sequence[str],
        run_metadata: Dict[str, object],
    ) -> None:
        self.args = args
        self.model = model
        self.model_avg = model_avg
        self.baseline = baseline
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.loaders = {
            "unsafe": endless(unsafe_loader),
            "safe": endless(safe_loader),
        }
        self.aligner = aligner
        self.output_dir = output_dir
        self.step = start_step
        self.lora_modules = list(lora_modules)
        self.run_metadata = run_metadata
        self.device = dist_util.dev()
        self.checked_gradients = False
        self.current_capture = (
            PreHeadCapture(model) if args.preservation_space == "hidden" else None
        )
        self.baseline_capture = (
            PreHeadCapture(baseline) if args.preservation_space == "hidden" else None
        )

    def _condition_with_decoupled_prefix(self, condition, motion):
        result = clone_tree(condition)
        result["y"]["prefix"] = motion[..., : self.args.context_len].detach()
        return result

    def _forward(self, model, motion, timesteps, condition, capture=None):
        if capture is None and model is self.model:
            capture = self.current_capture
        if capture is None and model is self.baseline:
            capture = self.baseline_capture
        if capture is not None:
            capture.value = None
        prediction = model(motion, timesteps, **clone_tree(condition))
        feature = capture.pop_bdt() if capture is not None else prediction
        return prediction, feature

    def _base_loss(self, prediction, target, mask):
        squared = (prediction - target).pow(2) * mask[:, None, None, :]
        return squared.sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0)

    def _weighted_kinematics(self, values):
        return (
            self.args.pose_weight * values["pose"]
            + self.args.velocity_weight * values["velocity"]
            + self.args.acceleration_weight * values["acceleration"]
        )

    def _unsafe_objective(self, motion, condition, timesteps, noise, prediction, mask):
        metrics = {}
        total = prediction.sum() * 0.0
        if self.args.harm_weight != 0.0:
            parts = kinematic_losses(
                prediction,
                motion,
                mask,
                self.args.frequency_mode,
                self.args.frequency_weight,
            )
            harm = self._weighted_kinematics(parts)
            metrics.update({"harm_" + key: value for key, value in parts.items()})
            if self.args.text_weight != 0.0:
                text = self.aligner.loss(
                    condition["y"]["word_embs"],
                    condition["y"]["pos_ohot"],
                    condition["y"]["sent_lens"],
                    prediction,
                    condition["y"]["lengths"],
                    self.args.text_temperature,
                )
                harm = harm + self.args.text_weight * text
                metrics["harm_text"] = text
            total = total + self.args.harm_weight * harm
            metrics["harm"] = harm
        if self.args.decouple_weight != 0.0:
            decoupled = decouple_motion(
                motion, condition["y"]["lengths"], self.args.decouple_segments
            )
            noisy = self.diffusion.q_sample(decoupled, timesteps, noise)
            dec_condition = self._condition_with_decoupled_prefix(condition, decoupled)
            prediction_dec, _ = self._forward(
                self.model, noisy, timesteps, dec_condition
            )
            parts = kinematic_losses(
                prediction_dec,
                decoupled,
                mask,
                self.args.frequency_mode,
                self.args.frequency_weight,
            )
            decouple = self._weighted_kinematics(parts)
            total = total + self.args.decouple_weight * decouple
            metrics.update({"decouple_" + key: value for key, value in parts.items()})
            metrics["decouple"] = decouple
        return total, metrics

    def _safe_objective(
        self, motion, condition, timesteps, noise, prediction, feature, mask
    ):
        metrics = {}
        total = prediction.sum() * 0.0
        if self.args.preservation_weight == 0.0:
            return total, metrics
        baseline_noise = noise
        if self.args.preservation_noise == "independent":
            baseline_noise = torch.randn_like(noise)
        baseline_noisy = self.diffusion.q_sample(motion, timesteps, baseline_noise)
        with torch.no_grad():
            _, baseline_feature = self._forward(
                self.baseline,
                baseline_noisy,
                timesteps,
                condition,
                self.baseline_capture,
            )
        current_pool = masked_pool(feature, mask)
        baseline_pool = masked_pool(baseline_feature, mask)
        main = -F.mse_loss(current_pool, baseline_pool)
        decouple = main.new_zeros(())
        if self.args.preservation_main_ratio < 1.0:
            decoupled = decouple_motion(
                motion, condition["y"]["lengths"], self.args.decouple_segments
            )
            noisy = self.diffusion.q_sample(decoupled, timesteps, noise)
            dec_condition = self._condition_with_decoupled_prefix(condition, decoupled)
            _, current_dec = self._forward(
                self.model,
                noisy,
                timesteps,
                dec_condition,
                self.current_capture,
            )
            baseline_condition = dec_condition
            if self.args.preservation_prefix == "legacy-original":
                baseline_condition = condition
            with torch.no_grad():
                _, baseline_dec = self._forward(
                    self.baseline,
                    noisy,
                    timesteps,
                    baseline_condition,
                    self.baseline_capture,
                )
            decouple = -F.mse_loss(
                masked_pool(current_dec, mask), masked_pool(baseline_dec, mask)
            )
        preservation = (
            self.args.preservation_main_ratio * main
            + (1.0 - self.args.preservation_main_ratio) * decouple
        )
        total = total + self.args.preservation_weight * preservation
        metrics.update(
            {
                "preservation_main": main,
                "preservation_decouple": decouple,
                "preservation": preservation,
            }
        )
        return total, metrics

    def update(self, stream: str):
        motion, condition = next(self.loaders[stream])
        motion = motion.to(self.device)
        condition = move_tree(condition, self.device)
        mask = condition["y"]["mask"].squeeze(1).squeeze(1).float()
        timesteps = torch.randint(
            0, self.diffusion.num_timesteps, (motion.shape[0],), device=self.device
        )
        noise = torch.randn_like(motion)
        noisy = self.diffusion.q_sample(motion, timesteps, noise)
        self.model.train()
        self.model.clip_model.eval()
        set_lora_scale(self.model, 1.0)
        self.optimizer.zero_grad(set_to_none=True)
        prediction, feature = self._forward(
            self.model,
            noisy,
            timesteps,
            condition,
            self.current_capture,
        )
        base = self._base_loss(prediction, motion, mask)
        if stream == "unsafe":
            objective, metrics = self._unsafe_objective(
                motion, condition, timesteps, noise, prediction, mask
            )
        else:
            objective, metrics = self._safe_objective(
                motion, condition, timesteps, noise, prediction, feature, mask
            )
        objective = objective + self.args.base_diffusion_weight * base
        metrics["base_diffusion"] = base
        metrics["objective"] = objective
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite objective at step {}".format(self.step + 1))
        objective.backward()
        if not self.checked_gradients:
            selected = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            missing = [name for name, parameter in selected if parameter.grad is None]
            if missing:
                raise RuntimeError("Selected LoRA tensors have no gradient: {}".format(missing))
            gradient_total = torch.stack(
                [parameter.grad.detach().abs().sum() for _, parameter in selected]
            ).sum()
            if not torch.isfinite(gradient_total) or gradient_total.item() == 0.0:
                raise RuntimeError("First LoRA update has no finite non-zero gradient")
            self.checked_gradients = True
        if self.args.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.args.gradient_clip,
            )
        self.optimizer.step()
        with torch.no_grad():
            averages = dict(self.model_avg.named_parameters())
            for name, current in self.model.named_parameters():
                if current.requires_grad:
                    averages[name].mul_(self.args.ema_decay).add_(
                        current, alpha=1.0 - self.args.ema_decay
                    )
        self.step += 1
        return {key: float(value.detach().item()) for key, value in metrics.items()}

    def save(self) -> Path:
        step_tag = "{:09d}".format(self.step)
        model_state = without_text_encoder(self.model.state_dict())
        average_state = without_text_encoder(self.model_avg.state_dict())
        metadata = {
            "format_version": 2,
            "step": self.step,
            "lora": {
                "modules": self.lora_modules,
                "rank": self.args.lora_rank,
                "alpha": self.args.lora_alpha,
                "dropout": self.args.lora_dropout,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": self.args.lr,
                "betas": [0.9, self.args.adam_beta2],
                "weight_decay": self.args.weight_decay,
            },
            "run": self.run_metadata,
        }
        checkpoint = self.output_dir / "model{}.pt".format(step_tag)
        atomic_torch(
            checkpoint,
            {
                "model": model_state,
                "model_avg": average_state,
                "step": self.step,
                "safemo_mmu": metadata,
            },
        )
        atomic_torch(
            self.output_dir / "lora_only_{}.pt".format(step_tag),
            {key: value for key, value in model_state.items() if key.endswith((".A", ".B"))},
        )
        atomic_torch(
            self.output_dir / "lora_only_avg_{}.pt".format(step_tag),
            {key: value for key, value in average_state.items() if key.endswith((".A", ".B"))},
        )
        atomic_torch(
            self.output_dir / "opt{}.pt".format(step_tag),
            {"step": self.step, "optimizer": self.optimizer.state_dict()},
        )
        atomic_json(
            self.output_dir / "latest.json",
            {"step": self.step, "checkpoint": checkpoint.name},
        )
        return checkpoint

    def run(self):
        log_path = self.output_dir / "train.jsonl"
        last_saved = None
        while self.step < self.args.steps:
            stream = "unsafe" if self.step % 2 == 0 else "safe"
            metrics = self.update(stream)
            record = {"step": self.step, "stream": stream, "time": time.time(), **metrics}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if self.step == 1 or self.step % self.args.log_interval == 0:
                print(
                    "step={} stream={} objective={:.6f}".format(
                        self.step, stream, metrics["objective"]
                    ),
                    flush=True,
                )
            if self.step % self.args.save_interval == 0:
                last_saved = self.save()
                print("saved {}".format(last_saved), flush=True)
        if self.step % self.args.save_interval != 0:
            last_saved = self.save()
            print("saved {}".format(last_saved), flush=True)
        return last_saved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train SafeMo Minimal Motion Unlearning")
    result.add_argument("--dependencies-root", required=True)
    result.add_argument("--humanml-root", required=True)
    result.add_argument("--unsafe-ids-file", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--base-checkpoint")
    result.add_argument("--base-component", choices=["model", "model_avg", "raw"], default="model_avg")
    result.add_argument("--bert-model")
    result.add_argument("--model-mean")
    result.add_argument("--model-std")
    result.add_argument("--train-split", default="train")
    result.add_argument("--steps", type=int, default=20000, help="Number of optimizer updates")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--device", default="0")
    result.add_argument("--seed", type=int, default=10)
    result.add_argument("--crop-policy", choices=["legacy-prefix", "random"], default="legacy-prefix")
    result.add_argument("--context-len", type=int, default=20)
    result.add_argument("--pred-len", type=int, default=40)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--adam-beta2", type=float, default=0.999)
    result.add_argument("--ema-decay", type=float, default=0.9999)
    result.add_argument("--gradient-clip", type=float, default=0.0)
    result.add_argument("--save-interval", type=int, default=1000)
    result.add_argument("--log-interval", type=int, default=200)
    result.add_argument("--resume")
    result.add_argument("--resume-optimizer")
    result.add_argument("--resume-without-optimizer", action="store_true")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--lora-rank", type=int, default=16)
    result.add_argument("--lora-alpha", type=float, default=16.0)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--lora-targets", default="ffn_in,ffn_out")
    result.add_argument("--harm-weight", type=float, default=2.5)
    result.add_argument("--decouple-weight", type=float, default=1.0)
    result.add_argument("--preservation-weight", type=float, default=0.5)
    result.add_argument("--base-diffusion-weight", type=float, default=0.0)
    result.add_argument("--pose-weight", type=float, default=0.35)
    result.add_argument("--velocity-weight", type=float, default=0.30)
    result.add_argument("--acceleration-weight", type=float, default=0.20)
    result.add_argument("--text-weight", type=float, default=0.10)
    result.add_argument("--text-temperature", type=float, default=0.07)
    result.add_argument("--decouple-segments", type=int, default=4)
    result.add_argument("--frequency-mode", choices=["none", "linear", "log"], default="none")
    result.add_argument("--frequency-weight", type=float, default=0.0)
    result.add_argument("--preservation-main-ratio", type=float, default=0.7)
    result.add_argument("--preservation-space", choices=["output", "hidden"], default="output")
    result.add_argument("--preservation-noise", choices=["independent", "shared"], default="independent")
    result.add_argument(
        "--preservation-prefix",
        choices=["legacy-original", "synchronized"],
        default="legacy-original",
    )
    return result


def validate(args) -> None:
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "lora_rank": args.lora_rank,
        "decouple_segments": args.decouple_segments,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("Positive integer arguments are required: {}".format(positive))
    if (
        args.workers < 0
        or args.context_len <= 0
        or args.pred_len <= 0
        or args.pred_len < args.context_len
    ):
        raise ValueError("Invalid worker count or motion lengths")
    scalars = [
        args.lr,
        args.weight_decay,
        args.ema_decay,
        args.lora_alpha,
        args.lora_dropout,
        args.gradient_clip,
        args.harm_weight,
        args.decouple_weight,
        args.preservation_weight,
        args.base_diffusion_weight,
        args.pose_weight,
        args.velocity_weight,
        args.acceleration_weight,
        args.text_weight,
        args.text_temperature,
        args.frequency_weight,
        args.preservation_main_ratio,
    ]
    if not all(math.isfinite(value) for value in scalars) or any(value < 0 for value in scalars):
        raise ValueError("Loss and optimizer scalars must be finite and non-negative")
    if not 0.0 <= args.preservation_main_ratio <= 1.0:
        raise ValueError("--preservation-main-ratio must be in [0, 1]")
    if not 0.0 <= args.ema_decay < 1.0 or not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("EMA decay and LoRA dropout must be in [0, 1)")
    if args.lr <= 0.0 or not 0.0 < args.adam_beta2 < 1.0:
        raise ValueError("Learning rate must be positive and Adam beta2 must be in (0, 1)")
    if args.lora_alpha <= 0.0:
        raise ValueError("--lora-alpha must be positive")
    if args.resume_optimizer and args.resume_without_optimizer:
        raise ValueError("Do not combine --resume-optimizer and --resume-without-optimizer")
    if args.text_weight > 0.0 and args.text_temperature <= 0.0:
        raise ValueError("--text-temperature must be positive when text loss is enabled")
    targets = [value.strip() for value in args.lora_targets.split(",") if value.strip()]
    if not targets or any(value not in {"ffn_in", "ffn_out"} for value in targets):
        raise ValueError("Clean training supports only effective targets: ffn_in,ffn_out")
    args.lora_targets_resolved = targets


def main() -> None:
    args = parser().parse_args()
    validate(args)
    dependencies = Path(args.dependencies_root).expanduser().resolve()
    humanml_root = Path(args.humanml_root).expanduser().resolve()
    unsafe_file = Path(args.unsafe_ids_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for label, path in (
        ("dependencies root", dependencies),
        ("HumanML3D root", humanml_root),
        ("unsafe IDs", unsafe_file),
    ):
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(label, path))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = first_file(
        args.base_checkpoint,
        [
            dependencies
            / "checkpoints/dip/DiP_no-target_10steps_context20_predict40/model000600343.pt"
        ],
        "DiP base checkpoint",
    )
    mean = first_file(
        args.model_mean,
        [dependencies / "data/dataset/HumanML3D/Mean.npy", humanml_root / "Mean.npy"],
        "HumanML3D Mean.npy",
    )
    std = first_file(
        args.model_std,
        [dependencies / "data/dataset/HumanML3D/Std.npy", humanml_root / "Std.npy"],
        "HumanML3D Std.npy",
    )
    glove = dependencies / "glove"
    for path in (
        glove / "our_vab_data.npy",
        glove / "our_vab_idx.pkl",
        glove / "our_vab_words.pkl",
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing GloVe dependency: {}".format(path))
    bert = args.bert_model or str(dependencies / "distilbert-base-uncased")
    bert_candidate = Path(bert).expanduser()
    bert = str(bert_candidate.resolve()) if bert_candidate.exists() else bert
    if args.text_weight > 0.0:
        evaluator = dependencies / "evaluators/t2m/text_mot_match/model/finest.tar"
        if not evaluator.is_file():
            raise FileNotFoundError("Missing text-motion evaluator: {}".format(evaluator))
    fixseed(args.seed)
    device_index = -1 if args.device.lower() == "cpu" else int(args.device)
    dist_util.setup_dist(device_index)
    device = dist_util.dev()
    hf_handler.get_dependencies = lambda: str(dependencies)
    from closd.diffusion_planner.model import mdm as mdm_module
    from closd.diffusion_planner.model.BERT.BERT_encoder import load_bert as real_load_bert

    mdm_module.load_bert = lambda _legacy_path: real_load_bert(bert)
    data_config = TrainDataConfig(
        humanml_root=humanml_root,
        unsafe_ids_file=unsafe_file,
        glove_dir=glove,
        mean=mean,
        std=std,
        split=args.train_split,
        context_len=args.context_len,
        pred_len=args.pred_len,
        crop_policy=args.crop_policy,
    )
    records, diagnostics = build_train_catalog(data_config)
    source_hashes = {
        "base_checkpoint_sha256": file_sha256(base_checkpoint),
        "unsafe_ids_sha256": file_sha256(unsafe_file),
        "humanml_mean_sha256": file_sha256(mean),
        "humanml_std_sha256": file_sha256(std),
    }
    unsafe_loader = build_train_loader(
        data_config, records, True, args.batch_size, args.workers, args.seed + 1
    )
    safe_loader = build_train_loader(
        data_config, records, False, args.batch_size, args.workers, args.seed + 2
    )
    model_args = main_table_args()
    model_args.context_len = args.context_len
    model_args.pred_len = args.pred_len
    bootstrap = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    model, diffusion = create_model_and_diffusion(model_args, bootstrap)
    base_info = load_backbone_checkpoint(model, base_checkpoint, args.base_component)
    replaced = inject_lora(
        model,
        args.lora_targets_resolved,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )
    if not replaced:
        raise RuntimeError("No LoRA layers matched")
    trainable = freeze_for_lora(model)
    baseline = copy.deepcopy(model).eval()
    model_avg = copy.deepcopy(model).eval()
    for frozen_model in (baseline, model_avg):
        for parameter in frozen_model.parameters():
            parameter.requires_grad = False
    start_step = 0
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, args.adam_beta2),
    )
    if args.resume:
        resume = Path(args.resume).expanduser().resolve()
        spec = checkpoint_lora_spec(resume, use_ema=False)
        if (
            spec["modules"] != sorted(replaced)
            or spec["rank"] != args.lora_rank
            or not math.isclose(float(spec["alpha"]), args.lora_alpha)
        ):
            raise RuntimeError("Resume checkpoint LoRA layout differs from this run")
        payload = torch.load(str(resume), map_location="cpu")
        saved_run = payload.get("safemo_mmu", {}).get("run", {})
        expected_resume = {
            **source_hashes,
            "base_component": base_info["component"],
        }
        mismatched = {
            key: {"checkpoint": saved_run.get(key), "current": value}
            for key, value in expected_resume.items()
            if saved_run.get(key) != value
        }
        saved_catalog = saved_run.get("data", {}).get("catalog_ids_sha256")
        if saved_catalog != diagnostics["catalog_ids_sha256"]:
            mismatched["catalog_ids_sha256"] = {
                "checkpoint": saved_catalog,
                "current": diagnostics["catalog_ids_sha256"],
            }
        if mismatched:
            raise RuntimeError("Resume inputs differ from the checkpoint: {}".format(mismatched))
        load_state(model, payload["model"], "resume model")
        load_state(model_avg, payload.get("model_avg", payload["model"]), "resume EMA")
        start_step = int(payload.get("step", 0))
        if not start_step:
            match = re.search(r"model(\d+)\.pt$", resume.name)
            start_step = int(match.group(1)) if match else 0
        if not args.resume_without_optimizer:
            optimizer_path = (
                Path(args.resume_optimizer).expanduser().resolve()
                if args.resume_optimizer
                else resume.with_name("opt{:09d}.pt".format(start_step))
            )
            if not optimizer_path.is_file():
                raise FileNotFoundError("Resume optimizer not found: {}".format(optimizer_path))
            optimizer_payload = torch.load(str(optimizer_path), map_location="cpu")
            optimizer.load_state_dict(optimizer_payload.get("optimizer", optimizer_payload))
            for group in optimizer.param_groups:
                group["lr"] = args.lr
                group["weight_decay"] = args.weight_decay
                group["betas"] = (0.9, args.adam_beta2)
    if start_step >= args.steps:
        raise ValueError("--steps must exceed resumed step {}".format(start_step))
    model.to(device)
    model_avg.to(device)
    baseline.to(device)
    move_optimizer(optimizer, device)
    model.clip_model.eval()
    model_avg.clip_model.eval()
    baseline.clip_model.eval()
    aligner = TextMotionAligner(device) if args.text_weight > 0.0 else None
    run_metadata = {
        **source_hashes,
        "base_component": base_info["component"],
        "data": diagnostics,
    }
    public_args = vars(args).copy()
    public_args["lora_targets_resolved"] = list(args.lora_targets_resolved)
    public_args.update(
        {
            "resolved_base_checkpoint": str(base_checkpoint),
            "resolved_humanml_root": str(humanml_root),
            "resolved_unsafe_ids_file": str(unsafe_file),
            "resolved_bert_model": bert,
            "resolved_device": str(device),
            "effective_lora_modules": replaced,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "start_step": start_step,
                "target_steps": args.steps,
                "lora_modules": len(replaced),
                "lora_tensors": len(trainable),
                "trainable_parameters": public_args["trainable_parameters"],
                "unsafe_samples": diagnostics["valid_unsafe"],
                "safe_samples": diagnostics["valid_safe"],
            },
            indent=2,
        )
    )
    if args.preflight_only:
        return
    atomic_json(output_dir / "args.json", public_args)
    atomic_json(output_dir / "data_diagnostics.json", diagnostics)
    trainer = Trainer(
        args,
        model,
        model_avg,
        baseline,
        diffusion,
        optimizer,
        unsafe_loader,
        safe_loader,
        aligner,
        output_dir,
        start_step,
        replaced,
        run_metadata,
    )
    trainer.run()


if __name__ == "__main__":
    main()
