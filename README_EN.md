<div align="center">
  <img width="40%" src="logo-Tengine.png">
  <h3> <a href="https://tengine-docs.readthedocs.io/en/latest/"> Documentation </a> | <a href="https://tengine-docs.readthedocs.io/zh_CN/latest/"> 中文文档 </a>  </h3>
</div>

English | [简体中文](./README.md)

# Tengine

[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](./LICENSE)
[![Build Status](https://drone.conleylee.com/api/badges/conley/Tengine/status.svg?ref=refs/heads/master)](https://drone.conleylee.com/conley/Tengine)
[![codecov](https://codecov.io/gh/ComingToy/Tengine/graph/badge.svg?token=KVOX0LW1NJ)](https://codecov.io/gh/ComingToy/Tengine)

## Introduction

**Tengine** is developed by **[OPEN AI LAB](http://www.openailab.com)**. This project meet the demand of **fast** and **efficient** deployment of deep learning neural network models on embedded devices. In order to achieve cross-platform deployment in many **AIoT** applications, this project is based on the original Tengine project using **C language** for reconstruction, and deep frame tailoring for the characteristics of limited embedded device resources. Also, it adopts a completely separated front-end/back-end design, which makes it possible to be transplanted and deployed onto CPU, GPU, NPU and other heterogeneous computing units rapidly, conveniently. At the same time, it is compatible with the original API and model format `tmfile` of **Tengine**, which reduces the cost of evaluation and migration.



The core code of Tengine Lite consists of 4 modules:

- [**device**](source/device): NN Operators back-end module, currently provides CPU code, and gradually open source GPU and NPU reference code;
- [**scheduler**](source/scheduler): core components of the framework, including NNIR, Computational Graphs, Hardware Resources, and the scheduling and execution modules of model serializer;
- [**operator**](source/operator): NN Operators front-end module, which realizes registration and initialization of NN Operators;
- [**serializer**](source/serializer): Model decoder, which decodes binary tmfile format into serialized model parameter.


## Architecture

![Tengine Architecture](doc/docs_en/images/architecture.png)


## How to use

### Compile

- [Quick Compilation](doc/docs_en/source_compile) Simple cross-platform compilation based on cmake.

### Example

- [examples](examples/) provides basic classification and detection algorithm use cases, which are continuously updated according to the needs of issues.

### Model Zoo

- [Baidu Netdisk](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) (password: 7ke5)

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)

### Model Convert tool

- [Pre-compiled version](https://github.com/OAID/Tengine/releases/download/lite-v1.2/convert_tool.zip): Pre-compiled model convert tool is provided on Linux system;
- [Online Convert tool](https://convertmodel.com/#outputFormat=tengine): Based on WebAssembly (the models are converted locally by browsers, no private data will be uploaded);
- [Source Compilation](https://github.com/OAID/Tengine-Convert-Tools): Refer to **Tengine-Convert-Tools** project, convert tool could be built by users.

### Speed assessment

- [Benchmark](benchmark/) Basic network speed assessment tool, any pull request is welcomed.

### NPU Plugin

- [TIM-VX](doc/docs_zh/source_compile/compile_timvx.md) VeriSilicon NPU user manual.

### AutoKernel Plugin

- [AutoKernel](https://github.com/OAID/AutoKernel.git).

### Container

- [SuperEdge](https://github.com/superedge/superedge) [How to use Tengine with SuperEdge](doc/docs_en/source_compile/deploy_SuperEdge.md).

## Roadmap

- [Road map](doc/docs_en/introduction/roadmap.md)

## Acknowledgement

Tengine Lite got ideas and developed based on these projects：

- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [MegEngine](https://github.com/MegEngine/MegEngine)
- [ONNX](https://github.com/onnx/onnx)
- [ncnn](https://github.com/Tencent/ncnn)
- [FeatherCNN](https://github.com/Tencent/FeatherCNN)
- [MNN](https://github.com/alibaba/MNN)
- [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [ACL](https://github.com/ARM-software/ComputeLibrary)
- [stb](https://github.com/nothings/stb)
- [convertmodel](https://convertmodel.com)
- [TIM-VX](https://github.com/VeriSilicon/TIM-VX)
- [SuperEdge](https://github.com/superedge/superedge)

## License

- [Apache 2.0](LICENSE)

## FAQ

- [FAQ common questions](doc/docs_en/introduction/faq.md)

## Tech Forum

- Github issues
- QQ groupchat: 829565581
- Email: Support@openailab.com
- Tengine Community: http://www.tengine.org.cn
