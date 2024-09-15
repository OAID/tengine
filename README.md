<div align="center">
  <img width="40%" src="logo-Tengine.png">
  <h3> <a href="https://tengine-docs.readthedocs.io/en/latest/"> Documentation </a> | <a href="https://tengine.readthedocs.io/zh_CN/latest/"> 中文文档 </a>  </h3>
</div>

简体中文 | [English](./README_EN.md)

# Tengine

[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](./LICENSE)
[![Build Status](https://drone.conleylee.com/api/badges/conley/Tengine/status.svg?ref=refs/heads/master)](https://drone.conleylee.com/conley/Tengine)
[![codecov](https://codecov.io/gh/ComingToy/Tengine/graph/badge.svg?token=KVOX0LW1NJ)](https://codecov.io/gh/ComingToy/Tengine)


## 简介

**Tengine** 由 **[OPEN AI LAB](http://www.openailab.com)** 主导开发，该项目实现了深度学习神经网络模型在嵌入式设备上的**快速**、**高效**部署需求。为实现在众多 **AIoT** 应用中的跨平台部署，本项目使用 **C 语言**进行核心模块开发，针对嵌入式设备资源有限的特点进行了深度框架裁剪。同时采用了完全分离的前后端设计，有利于 CPU、GPU、NPU 等异构计算单元的快速移植和部署，降低评估、迁移成本。

Tengine 核心代码由 4 个模块组成：

- [**device**](source/device)：NN Operators 后端模块，已提供 CPU、GPU、NPU 参考代码；
- [**scheduler**](source/scheduler)：框架核心部件，包括 NNIR、计算图、硬件资源、模型解析器的调度和执行模块；
- [**operator**](source/operator)：NN Operators 前端模块，实现 NN Operators 注册、初始化；
- [**serializer**](source/serializer)：模型解析器，实现 tmfile 格式的网络模型参数解析。


## 架构简析

![Tengine 架构](doc/docs_zh/images/architecture.png)

## 快速上手

### 编译

- [快速编译](doc/docs_zh/source_compile) 基于 cmake 实现简单的跨平台编译。

### 示例

- [examples](examples/) 提供基础的分类、检测算法用例，根据 issue 需求持续更新。
- [源安装](doc/docs_zh/quick_start/apt-get-install_user_manual.md) 提供ubuntu系统的apt-get命令行安装和试用，目前支持x86/A311D硬件。

### 模型仓库

- [百度网盘](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) （提取码：7ke5）

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)

### 转换工具

- [预编译版本](https://github.com/OAID/Tengine/releases/download/lite-v1.2/convert_tool.zip) ：提供 Ubuntu 18.04 系统上预编译好的模型转换工具；
- [在线转换版本](https://convertmodel.com/#outputFormat=tengine) ：基于 WebAssembly 实现（浏览器本地转换，模型不会上传；
- [源码编译](https://github.com/OAID/Tengine/tree/tengine-lite/tools/convert_tool) ：建议在服务器或者PC上编译，指令如下：
  ```
  mkdir build && cd build
  cmake -DTENGINE_BUILD_CONVERT_TOOL=ON ..
  make -j`nproc`
  ```

### 量化工具

- [源码编译](tools/quantize/README.md)：已开源量化工具源码，已支持 uint8/int8。

### 速度评估

- [Benchmark](benchmark/) 基础网络速度评估工具，欢迎大家更新。

### NPU Plugin

- [TIM-VX](doc/docs_zh/source_compile/compile_timvx.md) VeriSilicon NPU 使用指南。

### AutoKernel Plugin

- [AutoKernel](https://github.com/OAID/AutoKernel.git) 是一个简单易用，低门槛的自动算子优化工具，AutoKernel Plugin实现了自动优化算子一键部署到 Tengine 中。

### Container

- [SuperEdge](https://github.com/superedge/superedge) 借助 SuperEdge 边缘计算的开源容器管理系统，提供更便捷的业务管理方案；
- [How to use Tengine with SuperEdge](doc/docs_zh/source_compile/deploy_SuperEdge.md) 容器使用指南；
- [Video Capture user manual](doc/docs_zh/source_compile/demo_videocapture.md) Demo 依赖文件生成指南。

## Roadmap

- [Road map](doc/docs_zh/introduction/roadmap.md)

## 致谢

Tengine Lite 参考和借鉴了下列项目：

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

## 澄清说明

- [在线上报功能] 在线上报功能主要目的是了解Tengine的使用信息，信息用于优化和迭代Tengine，不会影响任何正常功能。该功能默认开启，如需关闭，可修改如下配置关闭：(主目录 CMakeLists.txt )  OPTION (TENGINE_ONLINE_REPORT "online report" OFF)

## FAQ

- [FAQ 常见问题](doc/docs_zh/introduction/faq.md)

## 技术讨论

- Github issues
- QQ 群: 829565581
- Email: Support@openailab.com
