# vllm/model_executor/models/beats.py
# Adds BEATs encoder support to vLLM

import os
import torch
import torch.nn as nn
# from vllm.model_executor.models.model import Model
from vllm.logger import init_logger

from beats.BEATs import BEATs, BEATsConfig

logger = init_logger(__name__)

class BeatsEncoderWrapper(nn.Module):  # <-- inherits from nn.Module
    def __init__(self, config):
        super().__init__()  # nn.Module init

        ckpt_path = config.get("beats_ckpt", None)
        if ckpt_path is None:
            raise ValueError("Missing 'beats_ckpt' path in config")

        logger.info(f"Loading BEATs checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(model_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: Tensor of shape [B, T] with 16kHz sampled audio
        Returns:
            features: Tensor of shape [B, T', D]
        """
        with torch.no_grad():
            out = self.model.extract_features(waveforms)

        return out[0] if isinstance(out, tuple) else out
