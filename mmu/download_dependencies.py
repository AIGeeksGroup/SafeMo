from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


root = Path("dependencies").resolve()
files = [
    "evaluators/t2m/text_mot_match/model/finest.tar",
    "glove/our_vab_data.npy",
    "glove/our_vab_idx.pkl",
    "glove/our_vab_words.pkl",
    "data/dataset/HumanML3D/Mean.npy",
    "data/dataset/HumanML3D/Std.npy",
    "checkpoints/dip/DiP_no-target_10steps_context20_predict40/args.json",
    "checkpoints/dip/DiP_no-target_10steps_context20_predict40/model000600343.pt",
]
for name in files:
    hf_hub_download(repo_id="guytevet/CLoSD", filename=name, local_dir=root)
snapshot_download(
    repo_id="distilbert/distilbert-base-uncased",
    local_dir=root / "distilbert-base-uncased",
)
print(root)
