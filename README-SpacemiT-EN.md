## Intruction
This repository is a RISC-V derivative version of SpacemiT on the ONNX Runtime community edition. Based on the RISC-V vector 1.0 instructions and SpacemiT AI extension instructions, some operators are optimized, while some defects within ONNXRuntime framework are fixed. The purpose of this open source repository is to enable a wider range of developers to understand and use the RISCV-Vector and IME instruction set, through the RVV and IME extensions, a typical AI application model can be used through this version of the ONNXRuntime to obtain a significant performance increase.

## Build
~~~ bash
# enter the onnxruntime build directory
bash tools/scripts/build_riscv64.spacemit.sh ${PWD} rv64 Release
~~~

## Build Options Description

The Spec definitions of SpacemiT AI extension instruction set is expanded instruction set in the build script and its Python build scripts, its naming convention is `spacemit-ime{version}`, a typical chip such as SpacemiT-K1 is equipped with 4-core A60 and 4-core X60, i.e., the first generation of IME extension `spacemit-ime1` exists on A60, and only `spacemit-ime1` is supported in the current version.

## Official Documentation and Forums

* If you want to get started faster, you can also refer to the official website of SpacemiT [AI Deployment Tool User Manual] (https://developer.spacemit.com/documentation?token=QfTKwODz3ifpHDkP5TbchWHBnJe)
* when problems encountered, you can go directly to SpacemiT [AI Forum] (https://forum.spacemit.com/c/ai/18e) to post an item.

