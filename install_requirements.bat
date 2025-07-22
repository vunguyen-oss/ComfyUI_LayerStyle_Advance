@echo off

set "python_exec=..\..\..\python_embeded\python.exe"
set "requirements_txt=%~dp0\requirements.txt"

echo Installing ComfyUI LayerStyle Advance dependencies
echo .
echo Installing special wheel files...
%python_exec% -s -m pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl
%python_exec% -s -m pip install ./whl/hydra_core-1.3.2-py3-none-any.whl

echo .
echo Uninstalling potentially conflicting packages...
%python_exec% -s -m pip uninstall -y onnxruntime
%python_exec% -s -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

echo .
echo Installing requirements...
for /f "delims=" %%i in (%requirements_txt%) do (
    %python_exec% -s -m pip install "%%i"
)

echo .
echo Installation complete!
pause

