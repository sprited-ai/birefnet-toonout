"""ToonOut on Replicate — anime-specialized background removal (matting).

ToonOut (arXiv:2509.06839, MIT) is a BiRefNet fine-tune by Muratori & Seytre
that fixes BiRefNet's weak spots on stylized content: hair wisps, line art,
translucency. Pixel accuracy 95.3% -> 99.5% on their anime test set.

Input: any image. Output: RGBA cutout (default) or the raw alpha matte.
"""

import pathlib

import torch
from cog import BasePredictor, Input, Path
from PIL import Image, ImageFilter
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
        # checkpoint was saved from a DDP + torch.compile wrapper
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v
                 for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        resolution: int = Input(
            description="Inference resolution (square). 1024 matches training.",
            default=1024, ge=256, le=2048,
        ),
        output_format: str = Input(
            description="rgba = cutout with alpha; mask = grayscale matte; color = composite on background_color",
            default="rgba", choices=["rgba", "mask", "color"],
        ),
        mask_blur: int = Input(
            description="Gaussian blur radius applied to the matte (softer edges)",
            default=0, ge=0, le=64,
        ),
        mask_offset: int = Input(
            description="Grow (+) or shrink (-) the matte by this many pixels",
            default=0, ge=-64, le=64,
        ),
        invert: bool = Input(
            description="Invert the matte (keep background instead of subject)",
            default=False,
        ),
        background_color: str = Input(
            description="Hex color for output_format=color, e.g. #222222",
            default="#222222",
        ),
    ) -> Path:
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

        if mask_offset > 0:
            for _ in range(mask_offset):
                matte = matte.filter(ImageFilter.MaxFilter(3))
        elif mask_offset < 0:
            for _ in range(-mask_offset):
                matte = matte.filter(ImageFilter.MinFilter(3))
        if mask_blur > 0:
            matte = matte.filter(ImageFilter.GaussianBlur(mask_blur))
        if invert:
            matte = Image.eval(matte, lambda v: 255 - v)

        out = pathlib.Path("/tmp/output.png")
        if output_format == "mask":
            matte.save(out)
        elif output_format == "color":
            bg = Image.new("RGB", src.size, background_color)
            bg.paste(src, (0, 0), matte)
            bg.save(out)
        else:
            rgba = src.convert("RGBA")
            rgba.putalpha(matte)
            rgba.save(out)
        return Path(out)
