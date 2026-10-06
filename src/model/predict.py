"""Load a trained checkpoint and reconstruct matching pairs of image points."""
from pathlib import Path

import numpy as np
import torch

from .network import ReconstructionModel


class ReconstructionPredictor:
    def __init__(self, checkpoint_path=None):
        path = Path(checkpoint_path) if checkpoint_path is not None else (
            Path(__file__).resolve().parent / "reconstruction.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Train the model first. Checkpoint missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model = ReconstructionModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.normalization = checkpoint["normalization"]
        for name, size in (("input_mean", 4), ("input_std", 4),
                           ("target_mean", 3), ("target_std", 3)):
            value = torch.as_tensor(self.normalization[name], dtype=torch.float32)
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid normalization value: {name}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"Normalization standard deviation must be positive: {name}")
            self.normalization[name] = value

    def predict(self, coordinates):
        """Accept (N, 4) raw pixel coordinates; return (N, 3) world points."""
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("Expected inputs shaped (N, 4): uA, vA, uB, vB")
        if not np.isfinite(coordinates).all():
            raise ValueError("Input coordinates must be finite")
        if len(coordinates) == 0:
            return np.empty((0, 3), dtype=np.float32)
        stats = self.normalization
        inputs = (torch.from_numpy(coordinates) - stats["input_mean"]) / stats["input_std"]
        with torch.no_grad():
            outputs = self.model(inputs)
            points = outputs * stats["target_std"] + stats["target_mean"]
        if not torch.isfinite(points).all():
            raise RuntimeError("The model produced nonfinite coordinates")
        return points.numpy()
