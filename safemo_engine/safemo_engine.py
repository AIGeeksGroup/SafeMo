#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
DEFAULT_MODEL = os.environ.get("SAFEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

LABEL_NAMES = {0: "safe", 1: "risky", 2: "harmful"}
LABEL_ALIASES = {
    "0": 0,
    "safe": 0,
    "1": 1,
    "risky": 1,
    "2": 2,
    "harmful": 2,
}
SPLIT_FILENAMES = {
    0: "level_1.txt",
    1: "level_2.txt",
    2: "level_3.txt",
}


def natural_text(text: str) -> str:
    return text.split("#", 1)[0].strip()


def read_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Prompt template not found: {path}") from exc


def parse_label(generated: str) -> int:
    text = generated.strip()
    direct = re.fullmatch(
        r"(?:(?:Output|Label|Answer)\s*:\s*)?([012])\s*[.!]?",
        text,
        flags=re.IGNORECASE,
    )
    if direct:
        return int(direct.group(1))

    standalone = re.findall(r"(?<![\w.+-])([012])(?!\w|\.\d)", text)
    if standalone and len(set(standalone)) == 1:
        return int(standalone[0])

    raise RuntimeError(
        "Classifier did not return one unambiguous label from 0, 1, or 2. "
        f"Raw generation: {generated!r}"
    )


def sentence_count(text: str) -> int:
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    return max(1, len(sentences))


def clean_rewrite(generated: str, original: str) -> str:
    text = generated.strip()
    text = re.sub(
        r"^(Output|Assistant|User|Input|Answer|Human|Level|Task|Rules|Examples)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    marker = re.search(
        r"(?:Output|Assistant|User|Input|Answer|Human|Level|Task|Rules|Examples)\s*:",
        text,
        flags=re.IGNORECASE,
    )
    if marker:
        text = text[: marker.start()].strip()

    boilerplate = re.search(
        r"\s+(?:"
        r"To continue following\b|"
        r"To maintain consistency(?: and clarity)?\b|"
        r"The output for\s+[\"']|"
        r"Here(?:'s| is) (?:the|a) (?:rewritten|refined|safe)\b|"
        r"Explanation\s*:|"
        r"Note\s*:"
        r")",
        text,
        flags=re.IGNORECASE,
    )
    if boilerplate:
        text = text[: boilerplate.start()].strip()

    text = text.splitlines()[0].strip() if text else ""
    text = re.sub(r"\s+", " ", text).strip()

    pieces = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(pieces[: sentence_count(original)]).strip()
    if not text:
        raise RuntimeError(f"Rewriter returned an empty result: {generated!r}")
    return text


class CausalLMRunner:
    def __init__(self, model_path: str, device_map: str, dtype: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing runtime dependencies. Create the project environment "
                "from environment.yml."
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[dtype],
            device_map=device_map,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        input_length = inputs["input_ids"].shape[-1]

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        new_tokens = output[0, input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class SafeMoEngine:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.classifier_prompt = read_prompt(PROMPT_DIR / "A1_temp.txt")
        self.risky_prompt = read_prompt(PROMPT_DIR / "A2_temp_lv2.txt")
        self.harmful_prompt = read_prompt(PROMPT_DIR / "A2_temp_lv3.txt")

    def classify(self, prompt: str) -> dict[str, Any]:
        text = prompt.strip()
        rendered = self.classifier_prompt.format(input_text=text)
        raw = self.runner.generate(rendered, max_new_tokens=8)
        label = parse_label(raw)
        return {
            "prompt": text,
            "label": label,
            "class": LABEL_NAMES[label],
            "dataset_level": label + 1,
        }

    def rewrite(self, prompt: str, label: int | None = None) -> dict[str, Any]:
        text = prompt.strip()
        if label is None:
            classification = self.classify(text)
            label = int(classification["label"])
        else:
            classification = {
                "prompt": text,
                "label": label,
                "class": LABEL_NAMES[label],
                "dataset_level": label + 1,
            }

        if label == 0:
            return {**classification, "rewritten": False, "text": text}

        template = self.risky_prompt if label == 1 else self.harmful_prompt
        rendered = template.format(input_text=text)
        raw = self.runner.generate(rendered, max_new_tokens=50)
        refined = clean_rewrite(raw, text)
        return {
            **classification,
            "rewritten": True,
            "text": refined,
        }


def first_prompt_from_file(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = natural_text(line)
            if text:
                return text
    return None


def read_split(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Required split file not found: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate sample identifier in split file: {path}")
    for name in names:
        validate_sample_name(name)
    return names


def read_checkpoint_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if content and not content.endswith("\n"):
        complete, separator, _ = content.rpartition("\n")
        recovered = complete + separator
        atomic_write_text(path, recovered)
    return read_split(path)


def validate_sample_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise SystemExit(
            "Sample identifiers must be extension-free basenames containing only "
            f"letters, numbers, underscores, or hyphens: {name!r}"
        )


def sample_path(root: Path, name: str) -> Path:
    validate_sample_name(name)
    path = root / f"{name}.txt"
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Sample path escapes its root directory: {path}") from exc
    return path


def atomic_write_text(path: Path, text: str) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def validate_rewrite_roots(
    texts_dir: Path,
    split_dir: Path,
    output_paths: dict[int, Path],
) -> None:
    for path in output_paths.values():
        if path.is_symlink():
            raise SystemExit(f"Refusing symlinked output directory: {path}")
        if path.exists() and not path.is_dir():
            raise SystemExit(f"Rewrite output is not a directory: {path}")
        if paths_overlap(path, texts_dir):
            raise SystemExit(
                f"Rewrite directory overlaps the source texts directory: {path}"
            )
        if paths_overlap(path, split_dir):
            raise SystemExit(f"Rewrite directory overlaps the split directory: {path}")


def write_names(path: Path, names: list[str]) -> None:
    atomic_write_text(path, "".join(f"{name}\n" for name in names))


def summary_text(split_dir: Path, counts: dict[int, int]) -> str:
    return (
        "=== A1 Summary ===\n"
        f"level_1.txt (label 0 - safe): {counts[0]} files\n"
        f"level_2.txt (label 1 - partly risky): {counts[1]} files\n"
        f"level_3.txt (label 2 - unsafe): {counts[2]} files\n"
        f"Lists saved to: {split_dir}\n"
    )


def finalize_splits(split_dir: Path, names_by_label: dict[int, list[str]]) -> None:
    for label, filename in SPLIT_FILENAMES.items():
        names_by_label[label].sort()
        write_names(split_dir / filename, names_by_label[label])

    unsafe_names = names_by_label[1] + names_by_label[2]
    write_names(split_dir / "unsafe.txt", unsafe_names)
    counts = {label: len(names) for label, names in names_by_label.items()}
    atomic_write_text(split_dir / "summary.txt", summary_text(split_dir, counts))


def classify_dataset(
    engine: SafeMoEngine,
    texts_dir: Path,
    split_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    if not texts_dir.is_dir():
        raise SystemExit(f"Texts directory not found: {texts_dir}")
    split_resolved = split_dir.resolve()
    texts_resolved = texts_dir.resolve()
    if split_resolved == texts_resolved or texts_resolved in split_resolved.parents:
        raise SystemExit("Split output must not be inside the source texts directory.")
    split_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        label: split_dir / filename for label, filename in SPLIT_FILENAMES.items()
    }
    for path in split_paths.values():
        if path.is_symlink():
            raise SystemExit(f"Refusing symlinked split file: {path}")
        if path.exists() and not path.is_file():
            raise SystemExit(f"Split output is not a file: {path}")

    populated = [
        path for path in split_paths.values() if path.exists() and path.stat().st_size
    ]
    if populated and not (resume or overwrite):
        raise SystemExit(
            f"Split output already contains results: {split_dir}. "
            "Use --resume or --overwrite."
        )

    names_by_label = {
        label: read_checkpoint_split(path) if resume else []
        for label, path in split_paths.items()
    }
    processed_labels: dict[str, int] = {}
    for label, names in names_by_label.items():
        for name in names:
            if name in processed_labels:
                raise SystemExit(
                    f"Duplicate sample identifier across split files: {name}"
                )
            processed_labels[name] = label

    mode = "a" if resume else "w"
    writers = {
        label: path.open(mode, encoding="utf-8") for label, path in split_paths.items()
    }
    text_files = sorted(texts_dir.glob("*.txt"), key=lambda path: path.name)
    new_count = 0
    skipped_empty = 0
    try:
        for scanned, path in enumerate(text_files, start=1):
            name = path.stem
            validate_sample_name(name)
            if name in processed_labels:
                continue

            prompt = first_prompt_from_file(path)
            if not prompt:
                skipped_empty += 1
                print(
                    f"[classify] empty file skipped: {path}",
                    file=sys.stderr,
                )
                continue

            result = engine.classify(prompt)
            label = int(result["label"])
            writers[label].write(name + "\n")
            writers[label].flush()
            os.fsync(writers[label].fileno())
            names_by_label[label].append(name)
            processed_labels[name] = label
            new_count += 1

            if progress_every > 0 and new_count % progress_every == 0:
                print(
                    f"[classify] {new_count} new files; "
                    f"{scanned}/{len(text_files)} scanned",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()
        finalize_splits(split_dir, names_by_label)

    return {
        "stage": "classify-dataset",
        "texts_dir": str(texts_dir),
        "split_dir": str(split_dir),
        "files_found": len(text_files),
        "newly_classified": new_count,
        "already_classified": len(processed_labels) - new_count,
        "empty_files_skipped": skipped_empty,
        "counts": {
            "level_1": len(names_by_label[0]),
            "level_2": len(names_by_label[1]),
            "level_3": len(names_by_label[2]),
            "unsafe": len(names_by_label[1]) + len(names_by_label[2]),
        },
    }


def rewrite_dataset(
    engine: SafeMoEngine,
    texts_dir: Path,
    split_dir: Path,
    output_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    if not texts_dir.is_dir():
        raise SystemExit(f"Texts directory not found: {texts_dir}")

    names_by_label = {
        1: read_split(split_dir / SPLIT_FILENAMES[1]),
        2: read_split(split_dir / SPLIT_FILENAMES[2]),
    }
    overlap = set(names_by_label[1]) & set(names_by_label[2])
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise SystemExit(f"level_2 and level_3 overlap: {examples}")

    output_paths = {
        1: output_dir / "level_2",
        2: output_dir / "level_3",
    }
    validate_rewrite_roots(texts_dir, split_dir, output_paths)
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.resolve().relative_to(output_dir.resolve())
        except ValueError as exc:
            raise SystemExit(
                f"Rewrite directory escapes its output root: {path}"
            ) from exc

    destinations = {
        label: {
            name: sample_path(output_paths[label], name)
            for name in names_by_label[label]
        }
        for label in (1, 2)
    }
    sources = {
        label: {name: sample_path(texts_dir, name) for name in names_by_label[label]}
        for label in (1, 2)
    }
    for label in (1, 2):
        for name in names_by_label[label]:
            if destinations[label][name].resolve() == sources[label][name].resolve():
                raise SystemExit(
                    f"Rewrite destination overlaps its source: {destinations[label][name]}"
                )
    expected_paths = {
        path for paths in destinations.values() for path in paths.values()
    }
    existing = [
        path for directory in output_paths.values() for path in directory.glob("*.txt")
    ]
    if existing and not (resume or overwrite):
        raise SystemExit(
            f"Rewrite output already contains results: {output_dir}. "
            "Use --resume or --overwrite."
        )
    stale = [path for path in existing if path not in expected_paths]
    if stale:
        examples = ", ".join(str(path) for path in sorted(stale)[:5])
        raise SystemExit(
            "Rewrite output contains files outside the current level splits. "
            f"Use a fresh output directory. Examples: {examples}"
        )

    total = sum(len(names) for names in names_by_label.values())
    rewritten = 0
    already_present = 0
    missing_sources = 0
    empty_sources = 0
    scanned = 0

    for label in (1, 2):
        for name in names_by_label[label]:
            scanned += 1
            destination = destinations[label][name]
            if destination.is_symlink():
                raise SystemExit(f"Refusing symlinked output file: {destination}")
            if destination.exists() and not destination.is_file():
                raise SystemExit(f"Rewrite output is not a file: {destination}")
            if resume and destination.exists() and destination.stat().st_size > 0:
                already_present += 1
                continue

            source = sources[label][name]
            if not source.exists():
                missing_sources += 1
                print(f"[rewrite] source not found: {source}", file=sys.stderr)
                continue

            prompt = first_prompt_from_file(source)
            if not prompt:
                empty_sources += 1
                print(
                    f"[rewrite] empty source skipped: {source}",
                    file=sys.stderr,
                )
                continue

            result = engine.rewrite(prompt, label=label)
            atomic_write_text(destination, str(result["text"]).strip() + "\n")
            rewritten += 1

            if progress_every > 0 and rewritten % progress_every == 0:
                print(
                    f"[rewrite] {rewritten} new files; {scanned}/{total} scanned",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "stage": "rewrite-dataset",
        "texts_dir": str(texts_dir),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "scheduled": total,
        "newly_rewritten": rewritten,
        "already_present": already_present,
        "missing_sources": missing_sources,
        "empty_sources": empty_sources,
    }


def publish_splits(source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked split directory: {destination_dir}")
    if destination_dir.exists() and not destination_dir.is_dir():
        raise SystemExit(f"Split output is not a directory: {destination_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    filenames = [*SPLIT_FILENAMES.values(), "unsafe.txt", "summary.txt"]
    for filename in filenames:
        source = source_dir / filename
        destination = destination_dir / filename
        if destination.is_symlink():
            raise SystemExit(f"Refusing symlinked split file: {destination}")
        if destination.exists() and not destination.is_file():
            raise SystemExit(f"Split output is not a file: {destination}")
        atomic_write_text(destination, source.read_text(encoding="utf-8"))


def validate_staging_directory(output_root: Path, staging_root: Path) -> None:
    if staging_root.name != ".safemo_engine_staging":
        raise RuntimeError(f"Unexpected staging path: {staging_root}")
    if staging_root.is_symlink():
        raise SystemExit(f"Refusing symlinked staging directory: {staging_root}")
    if staging_root.exists() and not staging_root.is_dir():
        raise SystemExit(f"Staging output is not a directory: {staging_root}")
    if staging_root.resolve().parent != output_root.resolve():
        raise SystemExit(f"Staging directory escapes its output root: {staging_root}")
    if not staging_root.exists():
        return

    entries = list(staging_root.iterdir())
    allowed_entries = {"A1_text_level", "state.json"}
    unexpected_entries = [
        path
        for path in entries
        if path.name not in allowed_entries
        and not (path.name.startswith(".state.json.") and path.name.endswith(".tmp"))
    ]
    if unexpected_entries:
        examples = ", ".join(str(path) for path in sorted(unexpected_entries)[:5])
        raise SystemExit(f"Unexpected files in staging directory: {examples}")

    state_temporary_files = [
        path
        for path in entries
        if path.name.startswith(".state.json.") and path.name.endswith(".tmp")
    ]
    invalid_state_temporary_files = [
        path
        for path in state_temporary_files
        if path.is_symlink() or not path.is_file()
    ]
    if invalid_state_temporary_files:
        examples = ", ".join(
            str(path) for path in sorted(invalid_state_temporary_files)[:5]
        )
        raise SystemExit(f"Invalid staging state temporary files: {examples}")

    state_path = staging_root / "state.json"
    if state_path.is_symlink():
        raise SystemExit(f"Refusing symlinked staging state: {state_path}")
    if state_path.exists() and not state_path.is_file():
        raise SystemExit(f"Staging state is not a file: {state_path}")

    staged_split_dir = staging_root / "A1_text_level"
    if staged_split_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked staging split: {staged_split_dir}")
    if staged_split_dir.exists() and not staged_split_dir.is_dir():
        raise SystemExit(f"Staging split is not a directory: {staged_split_dir}")
    if not staged_split_dir.exists():
        return

    allowed = {*SPLIT_FILENAMES.values(), "unsafe.txt", "summary.txt"}
    unexpected_files = [
        path
        for path in staged_split_dir.iterdir()
        if path.name not in allowed
        and not (path.name.startswith(".") and path.name.endswith(".tmp"))
    ]
    if unexpected_files:
        examples = ", ".join(str(path) for path in sorted(unexpected_files)[:5])
        raise SystemExit(f"Unexpected files in staging split: {examples}")
    symlinks = [path for path in staged_split_dir.iterdir() if path.is_symlink()]
    if symlinks:
        examples = ", ".join(str(path) for path in sorted(symlinks)[:5])
        raise SystemExit(f"Refusing symlinks in staging split: {examples}")
    non_files = [path for path in staged_split_dir.iterdir() if not path.is_file()]
    if non_files:
        examples = ", ".join(str(path) for path in sorted(non_files)[:5])
        raise SystemExit(f"Refusing non-files in staging split: {examples}")


def clear_staging_directory(output_root: Path, staging_root: Path) -> None:
    validate_staging_directory(output_root, staging_root)
    if staging_root.exists():
        shutil.rmtree(staging_root)


def read_staging_mode(staging_root: Path) -> str:
    state_path = staging_root / "state.json"
    candidate = state_path
    if not candidate.exists():
        candidates = sorted(staging_root.glob(".state.json.*.tmp"))
        if len(candidates) == 1:
            candidate = candidates[0]
    try:
        state = json.loads(candidate.read_text(encoding="utf-8"))
        mode = state["mode"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"Staging state is missing or invalid: {state_path}. Use --overwrite."
        ) from exc
    if mode not in {"fresh", "resume", "overwrite"}:
        raise SystemExit(f"Unknown staging mode in {state_path}: {mode!r}")
    return mode


def write_staging_mode(staging_root: Path, mode: str) -> None:
    staging_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        staging_root / "state.json",
        json.dumps({"mode": mode}, ensure_ascii=False) + "\n",
    )


def process_dataset(
    engine: SafeMoEngine,
    texts_dir: Path,
    output_root: Path,
    *,
    resume: bool,
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    if not texts_dir.is_dir():
        raise SystemExit(f"Texts directory not found: {texts_dir}")

    final_split_dir = output_root / "A1_text_level"
    final_refined_dir = output_root / "A2_texts_refined"
    final_output_paths = {
        1: final_refined_dir / "level_2",
        2: final_refined_dir / "level_3",
    }
    staging_root = output_root / ".safemo_engine_staging"
    staged_split_dir = staging_root / "A1_text_level"

    if final_split_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked split directory: {final_split_dir}")
    if final_split_dir.exists() and not final_split_dir.is_dir():
        raise SystemExit(f"Split output is not a directory: {final_split_dir}")
    split_resolved = final_split_dir.resolve()
    texts_resolved = texts_dir.resolve()
    if split_resolved == texts_resolved or texts_resolved in split_resolved.parents:
        raise SystemExit(
            "SafeMo Engine output must not be inside the source texts directory."
        )

    final_split_files = [
        final_split_dir / filename
        for filename in [*SPLIT_FILENAMES.values(), "unsafe.txt", "summary.txt"]
    ]
    for path in final_split_files:
        if path.is_symlink():
            raise SystemExit(f"Refusing symlinked split file: {path}")
        if path.exists() and not path.is_file():
            raise SystemExit(f"Split output is not a file: {path}")

    if final_split_dir.exists():
        allowed = {path.name for path in final_split_files}
        unexpected = [
            path for path in final_split_dir.glob("*.txt") if path.name not in allowed
        ]
        if unexpected:
            examples = ", ".join(str(path) for path in sorted(unexpected)[:5])
            raise SystemExit(
                "Split output contains unexpected files. "
                f"Use a fresh output directory. Examples: {examples}"
            )

    validate_rewrite_roots(texts_dir, final_split_dir, final_output_paths)
    existing_a1 = any(
        path.exists() and path.stat().st_size for path in final_split_files
    )
    existing_a2 = any(
        path
        for directory in final_output_paths.values()
        for path in directory.glob("*.txt")
    )
    if (existing_a1 or existing_a2) and not (resume or overwrite):
        raise SystemExit(
            f"SafeMo Engine output already contains results: {output_root}. "
            "Use --resume, --overwrite, or a fresh output directory."
        )

    validate_staging_directory(output_root, staging_root)
    if overwrite:
        clear_staging_directory(output_root, staging_root)
        transaction_mode = "overwrite"
        write_staging_mode(staging_root, transaction_mode)
    elif staging_root.exists() and any(staging_root.iterdir()):
        if not resume:
            raise SystemExit(
                f"An interrupted process-dataset run exists: {staging_root}. "
                "Use --resume or --overwrite."
            )
        transaction_mode = read_staging_mode(staging_root)
    else:
        transaction_mode = "resume" if resume else "fresh"
        write_staging_mode(staging_root, transaction_mode)

    staged_split_exists = staged_split_dir.exists()
    staged_split_dir.mkdir(parents=True, exist_ok=True)
    if transaction_mode == "resume" and not staged_split_exists:
        for filename in SPLIT_FILENAMES.values():
            source = final_split_dir / filename
            if source.exists():
                write_names(
                    staged_split_dir / filename,
                    read_checkpoint_split(source),
                )

    classification = classify_dataset(
        engine,
        texts_dir,
        staged_split_dir,
        resume=resume,
        overwrite=False,
        progress_every=progress_every,
    )
    counts = classification["counts"]
    atomic_write_text(
        staged_split_dir / "summary.txt",
        summary_text(
            final_split_dir,
            {
                0: int(counts["level_1"]),
                1: int(counts["level_2"]),
                2: int(counts["level_3"]),
            },
        ),
    )
    rewriting = rewrite_dataset(
        engine,
        texts_dir,
        staged_split_dir,
        final_refined_dir,
        resume=resume and transaction_mode != "overwrite",
        overwrite=transaction_mode == "overwrite",
        progress_every=progress_every,
    )
    publish_splits(staged_split_dir, final_split_dir)
    clear_staging_directory(output_root, staging_root)

    classification["split_dir"] = str(final_split_dir)
    rewriting["split_dir"] = str(final_split_dir)
    return {
        "stage": "process-dataset",
        "classification": classification,
        "rewriting": rewriting,
    }


def prompt_value(value: str | None) -> str:
    prompt = value if value is not None else sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("Provide --prompt TEXT or pipe a prompt on stdin.")
    return prompt


def label_value(value: str | None) -> int | None:
    if value is None:
        return None
    return LABEL_ALIASES[value.lower()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify and rewrite motion descriptions with SafeMo Engine."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="Local model path or Hugging Face model ID.",
        )
        subparser.add_argument(
            "--device-map",
            default="auto",
            help="Transformers device map (default: auto).",
        )
        subparser.add_argument(
            "--dtype",
            choices=("auto", "float16", "bfloat16", "float32"),
            default="float16",
            help="Model dtype (default: float16).",
        )

    def add_single_prompt_args(subparser: argparse.ArgumentParser) -> None:
        add_model_args(subparser)
        subparser.add_argument(
            "--prompt", help="Motion description; stdin is used if omitted."
        )

    def add_dataset_args(subparser: argparse.ArgumentParser) -> None:
        add_model_args(subparser)
        mode = subparser.add_mutually_exclusive_group()
        mode.add_argument(
            "--resume",
            action="store_true",
            help="Continue from existing outputs.",
        )
        mode.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite matching outputs.",
        )
        subparser.add_argument(
            "--progress-every",
            type=int,
            default=100,
            help="Report progress after this many new files; 0 disables it.",
        )

    classify_parser = subparsers.add_parser(
        "classify", help="Classify one motion description."
    )
    add_single_prompt_args(classify_parser)

    rewrite_parser = subparsers.add_parser(
        "rewrite", help="Classify and conditionally rewrite one description."
    )
    add_single_prompt_args(rewrite_parser)
    rewrite_parser.add_argument(
        "--label",
        choices=tuple(LABEL_ALIASES),
        help="Known label; skips classification when supplied.",
    )

    classify_dataset_parser = subparsers.add_parser(
        "classify-dataset", help="Create A1 split files for a texts directory."
    )
    add_dataset_args(classify_dataset_parser)
    classify_dataset_parser.add_argument("--texts-dir", required=True)
    classify_dataset_parser.add_argument("--split-dir", required=True)

    rewrite_dataset_parser = subparsers.add_parser(
        "rewrite-dataset", help="Rewrite level-2 and level-3 split members."
    )
    add_dataset_args(rewrite_dataset_parser)
    rewrite_dataset_parser.add_argument("--texts-dir", required=True)
    rewrite_dataset_parser.add_argument("--split-dir", required=True)
    rewrite_dataset_parser.add_argument("--output-dir", required=True)

    process_dataset_parser = subparsers.add_parser(
        "process-dataset", help="Classify and rewrite a texts directory."
    )
    add_dataset_args(process_dataset_parser)
    process_dataset_parser.add_argument("--texts-dir", required=True)
    process_dataset_parser.add_argument("--output-root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = CausalLMRunner(args.model, args.device_map, args.dtype)
    engine = SafeMoEngine(runner)

    if args.command == "classify":
        result = engine.classify(prompt_value(args.prompt))
    elif args.command == "rewrite":
        result = engine.rewrite(
            prompt_value(args.prompt), label=label_value(args.label)
        )
    elif args.command == "classify-dataset":
        result = classify_dataset(
            engine,
            Path(args.texts_dir).expanduser(),
            Path(args.split_dir).expanduser(),
            resume=args.resume,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
    elif args.command == "rewrite-dataset":
        result = rewrite_dataset(
            engine,
            Path(args.texts_dir).expanduser(),
            Path(args.split_dir).expanduser(),
            Path(args.output_dir).expanduser(),
            resume=args.resume,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
    else:
        result = process_dataset(
            engine,
            Path(args.texts_dir).expanduser(),
            Path(args.output_root).expanduser(),
            resume=args.resume,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
