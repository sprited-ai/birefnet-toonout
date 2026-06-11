"""One-time ToonOut → ONNX export.

Loads the local BiRefNet code + toonout safetensors (already on disk from the
comfy pipeline), wraps forward to emit the final sigmoid matte, exports
opset-17 ONNX at 1024x1024, then parity-checks ONNX vs torch on random input.

Run with the comfy venv python (has torch+transformers):
  .../comfy/.venv/bin/python export_onnx.py
"""

import sys

import numpy as np
import torch

RMBG_DIR = "/Users/jin/dev/sprite-dx/comfy/ComfyUI/models/RMBG/BiRefNet"
WEIGHTS = f"{RMBG_DIR}/BiRefNet_toonout.safetensors"
OUT = "/Users/jin/dev/birefnet-toonout/toonout-1024.onnx"
RES = 1024

sys.path.insert(0, RMBG_DIR)
from BiRefNet_config import BiRefNetConfig  # noqa: E402
from birefnet import BiRefNet  # noqa: E402

from safetensors.torch import load_file  # noqa: E402


class MatteWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[-1].sigmoid()


def main() -> None:
    print("loading model...", flush=True)
    model = BiRefNet(BiRefNetConfig(bb_pretrained=False))
    state = load_file(WEIGHTS)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"state dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  missing (first 5):", missing[:5], flush=True)
    model.eval()
    wrapped = MatteWrapper(model)

    dummy = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        ref = wrapped(dummy)
    print("torch forward ok, output", tuple(ref.shape), flush=True)

    print("exporting onnx...", flush=True)
    torch.onnx.export(
        wrapped, dummy, OUT,
        input_names=["image"], output_names=["matte"],
        opset_version=17, do_constant_folding=True,
        dynamo=False,
    )

    print("verifying with onnxruntime...", flush=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": dummy.numpy()})[0]
    diff = float(np.abs(out - ref.numpy()).max())
    print(f"parity max-abs-diff vs torch: {diff:.6f}", flush=True)
    import os
    print(f"done: {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
