// Copyright (c) 2023 SpacemiT. All rights reserved.
#include <set>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <dlfcn.h>
#include <core/common/common.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/framework/session_options.h"
#include "test_spacemit_ep.h"

namespace onnxruntime {

class Env;
namespace common {
class Status;
}

namespace test{
OrtStatus *InitSpaceMITExecutionProvider(Ort::SessionOptions& options,
                          const std::unordered_map<std::string, std::string> provider_options) {
  auto num_entries = provider_options.size();
  std::vector<const char*> keys, values;
  if (num_entries > 0) {
    keys.reserve(num_entries);
    values.reserve(num_entries);

    for (const auto& entry : provider_options) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }
  void* handle;
  auto error = Env::Default().LoadDynamicLibrary("libspacemit_ep.so", false, &handle);
  if (!error.IsOK()) {
    throw std::runtime_error(error.ErrorMessage());
  }
  OrtStatus* (*ep_init)(OrtSessionOptions*, const char* const*, const char* const*, size_t);
  error = Env::Default().GetSymbolFromLibrary(handle, "OrtSessionOptionsSpaceMITEnvInit", (void**)&ep_init);
  if (!error.IsOK()) {
    throw std::runtime_error(error.ErrorMessage());
  }
  auto sts = ep_init(options, keys.data(), values.data(), num_entries);
  return sts;
}
}
}