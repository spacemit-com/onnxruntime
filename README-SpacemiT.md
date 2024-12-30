## 简介
本仓库是进迭时空(SpacemiT)在ONNXRuntime社区版本的RISCV衍生版本, 基于RVV1.0及进迭时空AI扩展指令对部分算子进行了优化, 同时也对ONNXRuntime框架内的缺陷进行了修正. 本仓库开源的目的是能够让更广大的开发者了解并使用RISCV-Vector及进迭时空扩展指令集, 通过RVV及IME扩展, 典型的AI应用模型可通过该版本的ONNXRuntime获得极大性能提升.

## 编译构建
~~~ bash
# 进入onnxruntime目录执行脚本
bash tools/scripts/build_riscv64.spacemit.sh ${PWD} rv64 Release
~~~

## 编译选项说明
在构建脚本及其Python构建脚本中, 拓展了进迭时空AI扩展指令集的Spec定义, 其命名规则为`spacemit-ime{version}`, 典型芯片如SpacemiT-K1中搭载了4核A60及4核X60, 即A60上存在`spacemit-ime1`第一代IME扩展, 目前的版本中仅支持`spacemit-ime1`