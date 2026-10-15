# Audit sheet

## What it is

ComfyUI nodes that turn a plate sequence and its tracked USD camera into per-frame
point clouds, using the MoGe-3 depth model. A Houdini asset bakes them to BGEO for FX
holdouts. Runs fully on-prem.

## Model

| | |
|---|---|
| Model | MoGe-3 ViT-L, https://huggingface.co/Ruicheng/moge-3-vitl |
| Licence | MIT |
| Provenance | Linked as the official MoGe-3 checkpoint from https://github.com/microsoft/MoGe |
| File | `model.pt`, revision `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be`, 1,481,333,394 bytes |
| SHA-256 | `9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925` |
| Install to | ComfyUI `models/image2point/` |

## Source

| | |
|---|---|
| This project | https://github.com/brokenCushion/image2point, MIT |
| MoGe code (vendored in the release ZIP) | https://github.com/microsoft/MoGe, commit `74fbce054ebed49800de42d0ad0e83495065719a`, MIT (DINOv2 part Apache-2.0) |

## Security

- **No network access.** Nothing is downloaded at run time. Tests run the nodes with all
  network access blocked.
- **The model file cannot run code.** It is loaded with PyTorch `weights_only=True`.
- **Only MoGe's inference code** is vendored; training and download code are left out.
