#!/bin/bash

# Path to Python in virtual environment
PYTHON_EXEC="../../venv/bin/python"
REQUIREMENTS_TXT="./requirements.txt"

echo "Installing ComfyUI LayerStyle Advance dependencies"
echo ""
echo "Installing special wheel files..."
$PYTHON_EXEC -m pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl
$PYTHON_EXEC -m pip install ./whl/hydra_core-1.3.2-py3-none-any.whl

echo ""
echo "Uninstalling potentially conflicting packages..."
$PYTHON_EXEC -m pip uninstall -y onnxruntime
$PYTHON_EXEC -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

echo ""
echo "Installing requirements..."
while read requirement; do
    if [ ! -z "$requirement" ]; then
        $PYTHON_EXEC -m pip install "$requirement"
    fi
done < $REQUIREMENTS_TXT

echo ""
echo "Installation complete!" 