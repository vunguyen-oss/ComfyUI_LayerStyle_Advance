# ComfyUI with LayerStyle Advance - Setup Instructions

## Repositories to Clone
- Main ComfyUI repository: https://github.com/comfyanonymous/ComfyUI
- LayerStyle Advance extension: https://github.com/chflame163/ComfyUI_LayerStyle_Advance

## Python Dependencies
- Install requirements from a single requirements.txt file
- Special wheel files that need to be installed first:
  - docopt-0.6.2-py2.py3-none-any.whl
  - hydra_core-1.3.2-py3-none-any.whl
- Remove potentially conflicting packages before installation:
  - onnxruntime
  - opencv-python and variants

## Model Files Required
- Download models from:
  - https://huggingface.co/chflame163/ComfyUI_LayerStyle
  - Place in appropriate folders under ComfyUI/models

## Launch Instructions
- Run ComfyUI with appropriate parameters (--listen, --share) for remote access
- Include status updates for each step

## Additional Configuration
- Handle potential environment-specific issues (Google Colab, Kaggle)
- Set up appropriate paths
- Check for GPU availability 