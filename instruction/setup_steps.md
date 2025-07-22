# ComfyUI with LayerStyle Advance - Setup Instructions

## Repositories to Clone
- Main ComfyUI repository: https://github.com/comfyanonymous/ComfyUI
- LayerStyle Advance extension: https://github.com/chflame163/ComfyUI_LayerStyle_Advance

## Python Dependencies

### Windows Installation
- Install requirements from a single requirements.txt file
- Special wheel files that need to be installed first:
  - docopt-0.6.2-py2.py3-none-any.whl
  - hydra_core-1.3.2-py3-none-any.whl
- Remove potentially conflicting packages before installation:
  - onnxruntime
  - opencv-python and variants

### Linux Installation
- Use the provided install_requirements_linux.sh script
- May need to modify the Python path in the script to match your environment
- Ensure the script has executable permissions: `chmod +x install_requirements_linux.sh`
- Alternative: Manually install dependencies following the steps in linux_installation_guide.md

### Colab/Kaggle Installation
- Consider using a Python script for automated setup
- Modify Python paths to use the appropriate interpreter (e.g., `!pip install` instead of paths)
- Handle model downloads programmatically when possible

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