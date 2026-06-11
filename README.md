# birefnet-toonout

> **Unofficial community packaging.** This is not our model and we are not
> affiliated with its authors. ToonOut is the work of **Matteo Muratori and
> Joël Seytre** ([paper](https://arxiv.org/abs/2509.06839), MIT). We packaged
> it for [Replicate](https://replicate.com) because no hosted endpoint
> existed; **we earn nothing from it** — compute fees go to Replicate. If you
> are the authors and want this changed, transferred, or taken down:
> [open an issue](https://github.com/sprited-ai/birefnet-toonout/issues) and
> we will comply immediately.

[ToonOut](https://arxiv.org/abs/2509.06839) — anime-specialized background
removal — packaged for [Replicate](https://replicate.com).

BiRefNet is excellent on photos but stumbles on stylized content (hair wisps,
line art, translucent wings). ToonOut, a BiRefNet fine-tune by Matteo Muratori
and Joël Seytre on 1,228 curated anime images, lifts pixel accuracy from
95.3% to **99.5%** on their anime test set. Weights are MIT
([joelseytre/toonout](https://huggingface.co/joelseytre/toonout)).

- Input: any image (character art, generated sprites, anime frames)
- Output: RGBA cutout, or the raw alpha matte (`output_format: mask`)
- `resolution` 1024 matches training; raise for very large inputs

Packaged by [Sprited](https://spritedx.com) for the
[xsprite](https://github.com/sprited-ai/xsprite) sprite-generation workflow.

## Deploy

```
cog login && cog push r8.im/sprited/birefnet-toonout
```

## Citation

```bibtex
@misc{muratori2025toonout,
  title={ToonOut: Fine-tuned Background Removal for Anime Characters},
  author={Muratori, Matteo and Seytre, Joël},
  year={2025},
  eprint={2509.06839},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
