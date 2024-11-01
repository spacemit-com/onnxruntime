// Copyright (c) 2023 SpacemiT. All rights reserved.

#pragma once
#include <string>
#include "onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {
OrtStatus *InitSpaceMITExecutionProvider(Ort::SessionOptions& options,
                          const std::unordered_map<std::string, std::string> provider_options = {});
}
}