# ComfyUI Layer Style Advance

[English Readme Click Here](./README.md)  

从 [ComfyUI Layer Style](https://github.com/chflame163/ComfyUI_LayerStyle) 分离出来的节点，主要是依赖包安装要求较复杂的节点。

## 示例工作流

在 ```workflow``` 文件夹下有一些JSON工作流文件，这些是如何在ComfyUI中使用这些节点的示例。

## 安装方法

### 安装插件

* 建议使用ComfyUI Manager安装。

* 或者在ComfyUI的插件目录中打开cmd窗口，如```ComfyUI\custom_nodes```，输入

  ```
  git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git
  ```

* 或者下载zip文件并解压，将生成的文件夹复制到```ComfyUI\custom_nodes```中

### 安装依赖包

#### Windows安装

* 双击插件目录中的```install_requirements.bat```，等待安装完成。

* 或者手动安装依赖包：

1. 首先安装特殊的wheel文件：
```
pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl
pip install ./whl/hydra_core-1.3.2-py3-none-any.whl
```

2. 卸载可能冲突的包：
```
pip uninstall -y onnxruntime
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
```

3. 安装其他依赖：
```
pip install -r requirements.txt
```

#### Linux安装

* 使用提供的Linux安装脚本：

1. 使脚本可执行：
```bash
chmod +x install_requirements_linux.sh
```

2. 运行脚本：
```bash
./install_requirements_linux.sh
```

> **注意**：您可能需要修改脚本中的Python路径（`PYTHON_EXEC="../../../venv/bin/python"`）以匹配您的特定环境。

* 或者按照类似于Windows安装的步骤手动安装依赖，但使用Linux的Python路径。

* 详细的Linux安装说明，请参阅[Linux安装指南](instruction/linux_installation_guide.md)。

* 重启ComfyUI。

### 下载模型文件

从 [百度网盘](https://pan.baidu.com/s/1T_uXMX3OKIWOJLPuLijrgA?pwd=1yye) 或 [夸克网盘](https://pan.quark.cn/s/4802d6bca7cb) 或 [huggingface.co/chflame163/ComfyUI_LayerStyle](https://huggingface.co/chflame163/ComfyUI_LayerStyle/tree/main) 下载所有文件并将它们复制到```ComfyUI\models```文件夹。这个链接提供了本插件所需的所有模型文件。
或者根据每个节点的说明下载模型文件。
一些名为"Ultra"的节点将使用vitmatte模型，从 [vitmatte模型](https://huggingface.co/hustvl/vitmatte-small-composition-1k/tree/main) 下载并复制到```ComfyUI/models/vitmatte```文件夹，它也包含在上面的下载链接中。

## 常见问题

如果节点无法正常加载或使用过程中出现错误，请查看ComfyUI终端窗口中的错误信息。以下是常见错误及其解决方案。

### 警告：找不到xxxx.ini，使用默认xxxx..

此警告消息表示找不到ini文件，不影响使用。如果您不想看到这些警告，请将插件目录中的所有```*.ini.example```文件修改为```*.ini```。

### ModuleNotFoundError: No module named 'psd_tools'

此错误是由于```psd_tools```未正确安装。

解决方案：

* 关闭ComfyUI并在插件目录中打开终端窗口，执行以下命令：
  ```../../../python_embeded/python.exe -s -m pip install psd_tools```
  如果在安装psd_tool时出现错误，如```ModuleNotFoundError: No module named 'docopt'```，请下载[docopt的whl](https://www.piwheels.org/project/docopt/)并手动安装。
  在终端窗口中执行以下命令：
  ```../../../python_embeded/python.exe -s -m pip install path/docopt-0.6.2-py2.py3-none-any.whl```，其中```path```是whl文件的路径名。

### Cannot import name 'guidedFilter' from 'cv2.ximgproc'

此错误是由于```opencv-contrib-python```包的版本不正确，或者此包被其他opencv包覆盖。

### NameError: name 'guidedFilter' is not defined

此问题的原因与上述相同。

### Cannot import name 'VitMatteImageProcessor' from 'transformers'

此错误是由于```transformers```包版本过低导致的。

### insightface Loading very slow

此错误是由于```protobuf```包版本过低导致的。

### onnxruntime::python::CreateExecutionProviderInstance CUDA_PATH is set but CUDA wasn't able to be loaded. Please install the correct version of CUDA and cuDNN as mentioned in the GPU requirements page

解决方案：
重新安装```onnxruntime```依赖包。

### Error loading model xxx: We couldn't connect to huggingface.co ...

检查网络环境。如果您在中国无法正常访问huggingface.co，请尝试修改huggingface_hub包以强制使用hf_mirror。

* 在```huggingface_hub```包的目录中找到```constants.py```（通常在虚拟环境路径中的```Lib/site packages/huggingface_hub```），
  在```import os```后添加一行
  
  ```
  os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
  ```

### ValueError: Trimap did not contain foreground values (xxxx...)

此错误是在使用```PyMatting```方法处理蒙版边缘时，蒙版区域过大或过小导致的。

解决方案：

* 请调整参数以更改蒙版的有效区域。或使用其他方法处理边缘。

### Requests.exceptions.ProxyError: HTTPSConnectionPool(xxxx...)

当出现此错误时，请检查网络环境。

### UnboundLocalError: local variable 'clip_processor' referenced before assignment
### UnboundLocalError: local variable 'text_model' referenced before assignment
如果在执行```JoyCaption2```节点时出现此错误，并且已确认模型文件已放置在正确的目录中，
请检查```transformers```依赖包版本是否至少为4.43.2或更高。
如果```transformers```版本高于或等于4.45.0，并且还有错误消息：
```
Error loading models: De️️scriptors cannot be created directly.                                                                                           
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.                                
......
```
请尝试将```protobuf```依赖包降级至3.20.3，或设置环境变量：```PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python```。

## 更新

**如果更新后依赖包出错，请双击插件文件夹中的```install_requirements.bat```（Windows）或运行```./install_requirements_linux.sh```（Linux）重新安装依赖包。

* 提交[SAM2UltraV2](SAM2UltraV2)和[LoadSAM2Model](LoadSAM2Model)节点，将SAM模型更改为外部输入，以在使用多个节点时节省资源。
* 提交[JoyCaptionBetaOne](JoyCaptionBetaOne)，[LoadJoyCaptionBeta1Model](LoadJoyCaptionBeta1Model)，[JoyCaptionBeta1ExtraOptions](JoyCaptionBeta1ExtraOptions)节点，使用JoyCaption Beta One模型生成提示词。
* 提交[SaveImagePLusV2](SaveImagePlusV2)节点，添加自定义文件名和设置图像的dpi。
* 提交[GeminiImageEdit](#GeminiImageEdit)节点，支持使用gemini-2.0-flash-exp-image-generation API进行图像编辑。
* 提交[GeminiV2](#GeminiV2)和[ObjectDetectorGeminiV2](#ObjectDetectorGeminiV2)节点，使用支持gemini-2.0-flash-exp和gemini-2.5-pro-exp-03-25模型的google-genai依赖包。
* 添加夸克网盘模型下载链接。
* 支持numpy 2.x依赖包。
* 提交[DeepseekAPI_V2](#DeepseekAPI_V2)节点，支持阿里云和火山引擎API。
* 提交[Collage](#Collage)节点将图像拼贴成一个。
* 提交[DeepSeekAPI](DeepSeekAPI)节点，使用DeepSeek API进行文本推理。
* 提交[SegmentAnythingUltraV3](#SegmentAnythingUltraV3)和[LoadSegmentAnythingModels](#LoadSegmentAnythingModels)节点，避免在使用多个SAM节点时重复加载模型。
* 提交[ZhipuGLM4](#ZhipuGLM4)和[ZhipuGLM4V](#ZhipuGLM4V)节点，使用Zhipu API进行文本和视觉推理。在当前的Zhipu模型中，GLM-4-Flash和glm-4v-flash模型是免费的。
在[https://bigmodel.cn/usercenter/proj-mgmt/apikeys](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)免费申请API密钥，在```zhipu_api_key=```中填写您的API密钥。
* 提交[Gemini](#Gemini)节点，使用Gemini API进行文本或视觉推理。
* 提交[ObjectDetectorGemini](#ObjectDetectorGemini)节点，使用Gemini API进行对象检测。
* 提交[DrawBBOXMaskV2](#DrawBBOXMaskV2)节点，可以绘制圆角矩形蒙版。
* 提交[SmolLM2](#SmolLM2)，[SmolVLM](#SmolVLM)，[LoadSmolLM2Model](#LoadSmolLM2Model)和[LoadSmolVLMModel](#LoadSmolVLMModel)节点，使用SMOL模型进行文本推理和图像识别。
从[百度网盘](https://pan.baidu.com/s/1_jeNosYdDqqHkzpnSNGfDQ?pwd=to5b)或[huggingface](https://huggingface.co/chflame163/ComfyUI_LayerStyle/tree/main/ComfyUI/models/smol)下载模型文件并复制到```ComfyUI/models/smol```文件夹。
* Florence2添加支持[gokaygokay/Florence-2-Flux-Large](https://huggingface.co/gokaygokay/Florence-2-Flux-Large)和[gokaygokay/Florence-2-Flux](https://huggingface.co/gokaygokay/Florence-2-Flux)模型，
从[百度网盘](https://pan.baidu.com/s/1wBwJZjgMUKt0zluLAetMOQ?pwd=d6fb)或[huggingface](https://huggingface.co/chflame163/ComfyUI_LayerStyle/tree/main/ComfyUI/models/florence2)下载Florence-2-Flux-Large和Florence-2-Flux文件夹并复制到```ComfyUI\models\florence2```文件夹。
* 从requirements.txt文件中删除[ObjectDetector YOLOWorld](#ObjectDetectorYOLOWorld)节点所需的依赖。要使用此节点，请手动安装依赖包。
* 从[ComfyUI Layer Style](https://github.com/chflame163/ComfyUI_LayerStyle)分离一些节点到此存储库。

## Star 记录 

[![Star History Chart](https://api.star-history.com/svg?repos=chflame163/ComfyUI_LayerStyle_Advance&type=Date)](https://star-history.com/#chflame163/ComfyUI_LayerStyle_Advance&Date)

##  声明
LayerStyle Advance节点遵照MIT开源协议，有部分功能代码和模型来自其他开源项目，感谢原作者。如果作为商业用途，请查阅原项目授权协议使用。
