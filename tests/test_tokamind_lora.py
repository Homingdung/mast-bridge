from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from mast_bridge.training.tokamind_lora import (
    LoRAConfig,
    inject_lora_backbone,
    rebase_input_linear,
    rebase_output_linear,
)


class TokamindLoRATests(unittest.TestCase):
    def test_input_scaler_rebase_preserves_projected_values(self) -> None:
        layer = nn.Linear(3, 2)
        source_mean = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
        source_std = np.asarray([2.0, 4.0, 0.25], dtype=np.float32)
        target_mean = np.asarray([-1.0, 3.0, 1.5], dtype=np.float32)
        target_std = np.asarray([0.5, 2.0, 0.75], dtype=np.float32)
        raw = torch.tensor([[2.0, 1.0, -0.5]], dtype=torch.float32)

        expected = layer(
            (raw - torch.from_numpy(source_mean)) / torch.from_numpy(source_std)
        ).detach()
        rebase_input_linear(
            layer,
            source_mean=source_mean,
            source_std=source_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        actual = layer(
            (raw - torch.from_numpy(target_mean)) / torch.from_numpy(target_std)
        ).detach()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_output_scaler_rebase_preserves_raw_values(self) -> None:
        layer = nn.Linear(3, 2)
        source_mean = np.asarray([10.0, -3.0], dtype=np.float32)
        source_std = np.asarray([2.0, 0.5], dtype=np.float32)
        target_mean = np.asarray([8.0, 1.0], dtype=np.float32)
        target_std = np.asarray([4.0, 2.0], dtype=np.float32)
        hidden = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float32)

        source_prediction = layer(hidden).detach()
        expected_raw = (
            source_prediction * torch.from_numpy(source_std)
            + torch.from_numpy(source_mean)
        )
        rebase_output_linear(
            layer,
            source_mean=source_mean,
            source_std=source_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        target_prediction = layer(hidden).detach()
        actual_raw = (
            target_prediction * torch.from_numpy(target_std)
            + torch.from_numpy(target_mean)
        )

        torch.testing.assert_close(actual_raw, expected_raw, rtol=1e-5, atol=1e-6)

    def test_injected_lora_is_zero_initialised_and_only_adapter_is_trainable(
        self,
    ) -> None:
        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layer = nn.TransformerEncoderLayer(
                    d_model=8,
                    nhead=2,
                    dim_feedforward=16,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return self.encoder(values)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = Backbone()
                self.head = nn.Linear(8, 1)

        torch.manual_seed(4)
        model = Model()
        model.eval()
        values = torch.randn(2, 3, 8)
        expected = model.backbone(values).detach()

        report = inject_lora_backbone(
            model,
            LoRAConfig(rank=2, alpha=4.0),
        )
        actual = model.backbone(values).detach()
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(report["target_count"], 8)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertFalse(model.head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
