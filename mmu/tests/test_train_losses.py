import unittest

import torch

from safemo_mmu.train_losses import decouple_motion, kinematic_losses, masked_pool


class TrainLossesTest(unittest.TestCase):
    def test_identical_motion_has_zero_kinematic_loss(self):
        motion = torch.randn(2, 263, 1, 40)
        mask = torch.ones(2, 40)
        losses = kinematic_losses(motion, motion, mask, "none", 0.0)
        for value in losses.values():
            self.assertEqual(float(value), 0.0)

    def test_corrected_frequency_loss_backpropagates(self):
        prediction = torch.randn(2, 263, 1, 40, requires_grad=True)
        target = torch.randn_like(prediction)
        mask = torch.ones(2, 40)
        loss = sum(kinematic_losses(prediction, target, mask, "log", 0.1).values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_decoupling_and_pool_shapes(self):
        motion = torch.randn(3, 263, 1, 40)
        lengths = torch.full((3,), 40, dtype=torch.long)
        mask = torch.ones(3, 40)
        self.assertEqual(decouple_motion(motion, lengths, 4).shape, motion.shape)
        self.assertEqual(masked_pool(motion, mask).shape, (3, 263))


if __name__ == "__main__":
    unittest.main()
