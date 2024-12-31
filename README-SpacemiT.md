## 简介
本仓库是进迭时空(SpacemiT)在ONNXRuntime社区版本上的RISCV衍生版本, 基于RVV1.0及进迭时空AI扩展指令对部分算子进行了优化, 同时也对ONNXRuntime框架内的缺陷进行了修正. 本仓库开源的目的是能够让更广大的开发者了解并使用RISCV-Vector及进迭时空扩展指令集, 通过RVV及IME扩展, 典型的AI应用模型可通过该版本的ONNXRuntime获得极大性能提升.

## 编译构建
~~~ bash
# 进入onnxruntime目录执行脚本
bash tools/scripts/build_riscv64.spacemit.sh ${PWD} rv64 Release
~~~

## 编译选项说明
在构建脚本及其Python构建脚本中, 拓展了进迭时空AI扩展指令集的Spec定义, 其命名规则为`spacemit-ime{version}`, 典型芯片如SpacemiT-K1中搭载了4核A60及4核X60, 即A60上存在`spacemit-ime1`第一代IME扩展, 目前的版本中仅支持`spacemit-ime1`

## 官方文档及论坛

* 想要更快的上手，也可以参考进迭时空官网[AI部署工具使用手册](https://developer.spacemit.com/documentation?token=QfTKwODz3ifpHDkP5TbchWHBnJe)
* 如果遇到问题，可以直接到进迭时空[AI论坛](https://forum.spacemit.com/c/ai/18e)发帖交流

