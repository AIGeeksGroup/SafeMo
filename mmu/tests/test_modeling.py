import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from safemo_mmu.modeling import checkpoint_lora_spec, freeze_for_lora, inject_named_lora


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


class ModelingTest(unittest.TestCase):
    def test_named_lora_checkpoint_roundtrip(self):
        model = ToyModel()
        modules = ["block.0", "block.2"]
        self.assertEqual(inject_named_lora(model, modules, 2, 3.0, 0.0), modules)
        trainable = freeze_for_lora(model)
        self.assertEqual(len(trainable), 4)
        metadata = {"lora": {"modules": modules, "rank": 2, "alpha": 3.0}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_avg": model.state_dict(), "safemo_mmu": metadata}, path)
            spec = checkpoint_lora_spec(path, True)
        self.assertEqual(spec["modules"], modules)
        self.assertEqual(spec["rank"], 2)
        self.assertEqual(spec["alpha"], 3.0)


if __name__ == "__main__":
    unittest.main()
