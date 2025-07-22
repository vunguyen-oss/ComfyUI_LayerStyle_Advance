# ComfyUI LayerStyle Advance - Linux Installation Guide

This guide provides instructions for installing the ComfyUI LayerStyle Advance custom node package on Linux systems with ComfyUI installed in a virtual environment.

## Prerequisites

- A working installation of ComfyUI
- A Python virtual environment for ComfyUI

## Installation Steps

### 1. Clone the Repository

Navigate to your ComfyUI custom nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
```

Clone the repository:

```bash
git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git
cd ComfyUI_LayerStyle_Advance
```

### 2. Install Dependencies

#### Option 1: Using the provided installation script

We've included a Linux installation script that targets a virtual environment. You may need to modify it if your Python path is different.

1. Make the script executable:
   ```bash
   chmod +x install_requirements_linux.sh
   ```

2. Run the script:
   ```bash
   ./install_requirements_linux.sh
   ```

#### Option 2: Manual installation

If you prefer manual installation or need to customize the process:

1. Install special wheel files:
   ```bash
   /path/to/venv/bin/python -m pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl
   /path/to/venv/bin/python -m pip install ./whl/hydra_core-1.3.2-py3-none-any.whl
   ```

2. Remove potentially conflicting packages:
   ```bash
   /path/to/venv/bin/python -m pip uninstall -y onnxruntime
   /path/to/venv/bin/python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
   ```

3. Install the requirements:
   ```bash
   /path/to/venv/bin/python -m pip install -r requirements.txt
   ```

### 3. Download Model Files

Download model files as needed for specific nodes:

```bash
# Download models to appropriate directories under:
/path/to/ComfyUI/models/
```

The specific model files you need will depend on which nodes you plan to use. Refer to the [main README](https://github.com/chflame163/ComfyUI_LayerStyle_Advance#download-model-files) for links to all required model files.

### 4. Restart ComfyUI

Once installation is complete, restart ComfyUI to load the new nodes.

## Troubleshooting

If you encounter errors:

1. Check the ComfyUI terminal output for specific error messages
2. For dependency issues, try manually installing the specific package that's reporting errors
3. Ensure your virtual environment's Python path is correctly specified in the installation script

For more detailed troubleshooting, refer to the [Common Issues](https://github.com/chflame163/ComfyUI_LayerStyle_Advance#common-issues) section in the main README. 