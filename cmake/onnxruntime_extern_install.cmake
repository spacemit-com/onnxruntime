if (onnxruntime_BUILD_SHARED_LIB)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime)
endif()
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/path_string.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/type_list.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/profiler.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/string_utils.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/graph/function_template.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/graph)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/graph/ort_format_load_options.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/graph)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/graph/op_identifier.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/graph)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/graph/graph_utils.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/graph)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/graph/model.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/graph)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/optimizer/initializer.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/optimizer)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/optimizer/graph_optimizer_registry.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/optimizer)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/platform/env.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/platform)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/platform/env_time.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/platform)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/platform/telemetry.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/platform)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/platform/path_lib.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/platform)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/external_data_loader.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/external_data_loader_manager.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/model_metadef_id_generator.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/allocator_utils.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/allocator_stats.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/data_transfer.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/tuning_context.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/tuning_results.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/session_options.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/callback.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/config_options.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/library_handles.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/prepacked_weights_container.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/prepacked_weights.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/sequential_execution_plan.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/session_state.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/stream_execution_context.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/allocation_planner.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/execution_plan_base.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/data_transfer_manager.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/execution_providers.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/device_stream_collection.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/execution_frame.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/iexecutor.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/node_index_info.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/ort_value_pattern_planner.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/mem_pattern_planner.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/mem_pattern.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/memory_info.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/feeds_fetches_manager.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/fuse_nodes_funcs.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/ex_lib_loader.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/kernel_registry_manager.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/kernel_type_str_resolver.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/ort_value_name_idx_map.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/tensorprotoutils.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/endian_utils.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/mem_buffer.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/tensor_external_data_info.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/transpose_helper.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/compute_capability.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/framework/ep_context_options.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/framework)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/safeint.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/common/flatbuffers.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/common)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/util/thread_utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/util)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/util/math.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/util)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/providers/common.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/session/ort_apis.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/session)
install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/session/abi_key_value_pairs.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/session)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime)

install(FILES ${PROJECT_SOURCE_DIR}/../onnxruntime/core/session/abi_session_options_impl.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/session)

install(DIRECTORY ${flatbuffers_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)
install(DIRECTORY ${onnx_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)
install(DIRECTORY ${onnx_BINARY_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)
install(DIRECTORY ${eigen_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)
install(DIRECTORY ${protobuf_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)
install(DIRECTORY ${protobuf_BINARY_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}/third_party)

if (onnxruntime_ENABLE_PYTHON)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
endif()

install(TARGETS onnxruntime_provider_test
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

if (onnxruntime_BUILD_BENCHMARKS)
install(TARGETS onnxruntime_perf_test
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS onnxruntime_mlas_test
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS onnxruntime_mlas_benchmark
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()