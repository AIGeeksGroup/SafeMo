# SafeMo: Linguistically Grounded Unlearning for Trustworthy Text-to-Motion Generation

This is the official repository for the paper:

> **SafeMo: Linguistically Grounded Unlearning for Trustworthy Text-to-Motion Generation**
>
> Yiling Wang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Yiran Wang, and [Hao Tang](https://ha0tang.github.io/)<sup>‡</sup>
>
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author.
>
> ***EMNLP 2026 Findings***
>
> ### [Paper](https://arxiv.org/abs/2601.00590) | [Website](https://aigeeksgroup.github.io/SafeMo/) | [Code](https://github.com/AIGeeksGroup/SafeMo) | [Model](https://huggingface.co/AIGeeksGroup/SafeMo) | [SafeMoVAE-29K](https://huggingface.co/datasets/AIGeeksGroup/SafeMoVAE-29K) | [SafeMoVQ-29K](https://huggingface.co/datasets/AIGeeksGroup/SafeMoVQ-29K)

## Environment
```bash
conda env create -f environment.yml
conda activate safemo
```

## SafeMoEngine

SafeMoEngine is the language-agent component of SafeMo. It assigns each motion description to one of three safety levels and conditionally rewrites
non-safe descriptions:

- level 1 (`label=0`, safe): retained without modification;
- level 2 (`label=1`, risky): the unsafe motion is locally refined;
- level 3 (`label=2`, harmful): the description is rewritten as a peaceful,
  non-contact motion.

The experiments used
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
`--model` accepts either a Hugging Face model identifier or a local checkpoint
directory:

```bash
export SAFEMO_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# Or use an existing checkpoint:
export SAFEMO_LLM_MODEL=/path/to/Qwen2.5-7B-Instruct
```

Other instruction-tuned causal language models compatible with Hugging Face
`AutoTokenizer` and `AutoModelForCausalLM` can also be supplied. Their outputs
may differ from the reference configuration.

### Inference

Classify one motion description:

```bash
python safemo_engine/safemo_engine.py classify \
  --model "$SAFEMO_LLM_MODEL" \
  --prompt "a person punches someone with his right fist"
```

Classify and conditionally rewrite one description:

```bash
python safemo_engine/safemo_engine.py rewrite \
  --model "$SAFEMO_LLM_MODEL" \
  --prompt "a person punches someone with his right fist"
```

Safe descriptions are returned unchanged. If the label is already known, the classification call can be skipped.

```bash
python safemo_engine/safemo_engine.py rewrite \
  --model "$SAFEMO_LLM_MODEL" \
  --label harmful \
  --prompt "a person punches someone with his right fist"
```

Each command emits one JSON object. A description can also be supplied through standard input by omitting `--prompt`.

### Dataset classify-then-rewrire processing

Classify and rewrite a complete `texts/` directory (e.g., HumanML3D texts) in one model-loading session:

```bash
python safemo_engine/safemo_engine.py process-dataset \
  --model "$SAFEMO_LLM_MODEL" \
  --texts-dir /path/to/texts \
  --output-root /path/to/safemo_engine_output
```

The output follows the structure used in the SafeMo experiments:

```text
safemo_engine_output/
├── A1_text_level/
│   ├── level_1.txt
│   ├── level_2.txt
│   ├── level_3.txt
│   ├── unsafe.txt
│   └── summary.txt
└── A2_texts_refined/
    ├── level_2/
    └── level_3/
```
In `A1_text_level`, each split contains one extension-free sample identifier per line.
`unsafe.txt` is the ordered concatenation of `level_2.txt` and `level_3.txt`.
The `A2_texts_refined` contain one rewritten caption file for each level-2
or level-3 sample.

Classification and rewriting can also be run separately:

```bash
python safemo_engine/safemo_engine.py classify-dataset \
  --model "$SAFEMO_LLM_MODEL" \
  --texts-dir /path/to/texts \
  --split-dir /path/to/safemo_engine_output/A1_text_level

python safemo_engine/safemo_engine.py rewrite-dataset \
  --model "$SAFEMO_LLM_MODEL" \
  --texts-dir /path/to/texts \
  --split-dir /path/to/safemo_engine_output/A1_text_level \
  --output-dir /path/to/safemo_engine_output/A2_texts_refined
```

Dataset commands use the first non-empty caption in each HumanML3D text file and remove its `#...` metadata. Use `--resume` to continue an interrupted run, or `--overwrite` to replace matching outputs. Neither mode deletes unrelated files. Recommend using a fresh output directory when split membership changes.


## Minimal Motion Unlearning

The SafeMo model applies MMU to a [CLoSD/DiP](https://github.com/GuyTevet/CLoSD) model.

### Prerequisites


```bash
export SAFEMO_ROOT=/path/to/SafeMo_Release
cd "$SAFEMO_ROOT/mmu"
python download_dependencies.py
```

Prepare [HumanML3D](https://github.com/EricGuo5513/HumanML3D) according to official instructions. The dataset folder should appear as follows:

```text
HumanML3D/
├── Mean.npy
├── Std.npy
├── all.txt
├── train.txt
├── val.txt
├── test.txt
├── texts/
└── new_joint_vecs/
```

Download the SafeMo checkpoint:

```bash
huggingface-cli download AIGeeksGroup/SafeMo_MMU model000020000.pt \
  --local-dir checkpoints/safemo-mmu
```

Optionally, you can set the paths once:

```bash
export CKPT=checkpoints/safemo-mmu/model000020000.pt
export DEPS=dependencies
export BERT=dependencies/distilbert-base-uncased
export HML=/path/to/HumanML3D
```

### Evaluation

The evaluation writes `summary.txt`, `summary.json`, `raw_metrics.json`, and `protocol.json`. The summary reports FID, Diversity, and R-precision. `--repetitions` defaults to 1, and `--seed` is required because generated metrics vary with sampling randomness. Each run requires a new `--output-dir`.

#### LCR versions and splits

The SafeMo experiments were conducted when only [Human Motion Unlearning v1](https://arxiv.org/abs/2503.18674v1) was public. Its code and unsafe IDs were not available, so we independently reimplemented its full-dataset HumanML3D split from the paper. A substantially revised version later appeared as [arXiv v3](https://arxiv.org/abs/2503.18674v3) and was published at [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37351), and the official implementation of the new version LCR is available. The two versions differ much, and only the later version has official implementation. We include both protocols so the original setting and the later setting can be evaluated separately.

- `splits/lcr_wholeset_reimplementation_unsafe.txt`: our reimplementation of the LCR v1 unsafe split over the full HumanML3D ID space, according to their paper.
- `splits/lcr_official_test_unsafe.txt`: the later LCR unsafe split generated using the later version's official implementation.
- `splits/safemo_unsafe.txt`: Unsafe splits classified by our SafeMoEngine classifier.

Evaluate the reimplemented LCR v1 full-dataset split with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m safemo_mmu.evaluate \
  --checkpoint "$CKPT" --dependencies-root "$DEPS" \
  --humanml-root "$HML" --bert-model "$BERT" \
  --unsafe-ids-file splits/lcr_wholeset_reimplementation_unsafe.txt \
  --dataset-scope all --seed 10 --repetitions 1 \
  --output-dir outputs/eval_lcr_reimplementation_all_seed10
```

Evaluate the later official LCR (v3) split on the HumanML3D test set with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m safemo_mmu.evaluate \
  --checkpoint "$CKPT" --dependencies-root "$DEPS" \
  --humanml-root "$HML" --bert-model "$BERT" \
  --unsafe-ids-file splits/lcr_official_test_unsafe.txt \
  --dataset-scope test --seed 10 --repetitions 1 \
  --output-dir outputs/eval_lcr_official_test_seed10
```

Evaluate the unsafe split classified by the SafeMo Engine with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m safemo_mmu.evaluate \
  --checkpoint "$CKPT" --dependencies-root "$DEPS" \
  --humanml-root "$HML" --bert-model "$BERT" \
  --unsafe-ids-file splits/safemo_unsafe.txt \
  --dataset-scope all --seed 10 --repetitions 1 \
  --output-dir outputs/eval_safemo_all_seed10
```

<details>
<summary>BibTeX for the two LCR versions</summary>

```bibtex
@misc{dematteis2025hmu_v1,
  title         = {Human Motion Unlearning},
  author        = {De Matteis, Edoardo and Migliarini, Matteo and Sampieri, Alessio and Spinelli, Indro and Galasso, Fabio},
  year          = {2025},
  eprint        = {2503.18674},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  version       = {1},
  note          = {arXiv version 1, 24 March 2025},
  url           = {https://arxiv.org/abs/2503.18674v1}
}

@article{dematteis2026hmu_aaai,
  title   = {Human Motion Unlearning},
  author  = {De Matteis, Edoardo and Migliarini, Matteo and Sampieri, Alessio and Spinelli, Indro and Galasso, Fabio},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {5},
  pages   = {3533--3541},
  year    = {2026},
  doi     = {10.1609/aaai.v40i5.37351},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/37351}
}
```

</details>

### Training

Train your own model using

```bash
python -m safemo_mmu.train \
  --dependencies-root "$DEPS" \
  --humanml-root "$HML" \
  --unsafe-ids-file splits/safemo_unsafe.txt \
  --output-dir /path/to/output \
  --steps 20000 \
  --seed [seed] \
  --harm-weight [harm-weight] \
  --decouple-weight [decouple-weight] \
  --preservation-weight [preservation-weight] \
  --base-diffusion-weight [base-diffusion-weight] \
  --pose-weight [pose-weight] \
  --velocity-weight [velocity-weight] \
  --acceleration-weight [acceleration-weight] \
  --text-weight [text-weight] \
  --text-temperature [text-temperature] \
  --frequency-mode none \
  --frequency-weight 0 \
  --preservation-main-ratio [preservation-main-ratio]
```

Training alternates between unsafe and safe batches. The command-line weights define the following objectives:

```text
L_unsafe = w_diff * L_diff
         + w_harm * (w_pose * L_pose + w_vel * L_vel
                    + w_acc * L_acc + w_text * L_text)
         + w_dec * (w_pose * L_pose_dec + w_vel * L_vel_dec
                   + w_acc * L_acc_dec)

L_safe   = w_diff * L_diff
         + w_pres * (r * L_pres_main + (1 - r) * L_pres_dec)
```

Here, `r` is `--preservation-main-ratio`, and the preservation terms are negative MSE distances from the frozen baseline. Weights remain fixed during a run; set any weight to zero to disable its term.

### Visualization

The required fitting and rendering code is included under `visualization/`. The SMPL model, mean parameters, and GMM pose prior are not redistributed. Obtain them under their respective licenses and place them as follows:

```text
visualization/deps/smpl_models/
├── SMPL_NEUTRAL.pkl
├── neutral_smpl_mean_params.h5
└── gmm_08.pkl
```

Download Blender 2.93.18 from the [Blender 2.93 LTS archive](https://www.blender.org/download/lts/2-93/), extract it, and verify the installation:

```bash
tar -xvf blender-2.93.18-linux-x64.tar.xz
cd blender-2.93.18-linux-x64
./blender --background --version
./blender --background --python-expr "import sys; print(sys.version.split(' ')[0])"
./blender --background --python-expr "import sys; print(sys.executable)"
```

Use the reported Blender Python path in place of `/path/to/blender-python`:

```bash
/path/to/blender-python -m ensurepip --upgrade
/path/to/blender-python -m pip install --upgrade pip
/path/to/blender-python -m pip install numpy==2.0.2 matplotlib==3.9.4 \
  hydra-core==1.3.2 hydra-colorlog==1.2.0 moviepy==1.0.3 \
  shortuuid==1.0.13 natsort==8.4.0 tqdm==4.67.1
```

Run the following commands from the vendored visualization directory. Each input must be a HumanML3D joint sequence with shape `(frames, 22, 3)`. Fitting writes a sibling `*_mesh.npy` file, which the renderer then converts to video:

```bash
cd "$SAFEMO_ROOT/visualization"
python -m fit --dir /path/to/joint_npy --save_folder /path/to/smplfit --gpu_ids 0
/path/to/blender --background --python render.py -- \
  --cfg=./configs/render_mld.yaml --dir=/path/to/joint_npy \
  --mode=video --joint_type=HumanML3D
```

Use `--mode=video` for MP4 output or `--mode=sequence` for a single PNG showing the motion sequence. SMPL/SMPL-X assets have separate licenses and are not redistributed here.
