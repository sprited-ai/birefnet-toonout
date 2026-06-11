"""ToonOut on Replicate — anime-specialized background removal (matting).

ToonOut (arXiv:2509.06839, MIT) is a BiRefNet fine-tune by Muratori & Seytre
that fixes BiRefNet's weak spots on stylized content: hair wisps, line art,
translucency. Pixel accuracy 95.3% -> 99.5% on their anime test set.

Input: any image. Output: RGBA cutout (default) or the raw alpha matte.
"""

from pathlib import Path

import torch
from cog import BasePredictor, Input
from cog import Path as CogPath
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

WEIGHTS = "/weights/birefnet_finetuned_toonout.pth"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def predict(
        self,
        image: CogPath = Input(description="Input image"),
        resolution: int = Input(
            description="Inference resolution (square). 1024 matches training.",
            default=1024, ge=256, le=2048,
        ),
        output_format: str = Input(
            description="rgba = cutout with alpha; mask = grayscale matte",
            default="rgba", choices=["rgba", "mask"],
        ),
    ) -> CogPath:
        src = Image.open(str(image)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        batch = tf(src).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            batch = batch.half()

        with torch.no_grad():
            preds = self.model(batch)[-1].sigmoid().float().cpu()
        matte = transforms.ToPILImage()(preds[0].squeeze()).resize(src.size)

        out = Path("/tmp/output.png")
        if output_format == "mask":
            matte.save(out)
        else:
            rgba = src.convert("RGBA")
            rgba.putalpha(matte)
            rgba.save(out)
        return CogPath(out)
