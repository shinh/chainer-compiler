get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_COMMON_DIR ${CHAINER_COMPILER_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)

if (${CHAINER_COMPILER_ENABLE_PYTHON})
  add_definitions(-DCHAINER_COMPILER_ENABLE_PYTHON=1)
endif()

add_definitions(-DCHAINER_COMPILER_DLDT_DIR=\"${CHAINER_COMPILER_DLDT_DIR}\")

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_node.py
  MAIN_DEPENDENCY gen_node.py
  DEPENDS ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py)
add_custom_target(gen_node_base_h DEPENDS gen_node_base.h)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.cc
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/chxvm/gen_chxvm_codegen.py
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY chxvm/gen_chxvm_codegen.py
  DEPENDS
  ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py
  ${CHAINER_COMPILER_ROOT_DIR}/runtime/chxvm_defs.py
  ${CHAINER_COMPILER_ROOT_DIR}/runtime/chxvm.proto.tmpl
  )
add_custom_target(
  compiler_chxvm_codegen_h
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.h
  )

add_custom_command(
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/flags.h"
  COMMAND
  ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode flags.h
  --output "${CMAKE_CURRENT_BINARY_DIR}/flags.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_command(
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  COMMAND
  ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode flags.cc
  --output "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_target(
  compiler_flags_h
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/flags.h" "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  )

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CHAINER_COMPILER_TVM_INCLUDE_DIRS})

add_library(chainer_compiler_compiler
  code_emitter.cc
  constant_propagation.cc
  computation_order/core.cc
  computation_order/policy_chen.cc
  computation_order/policy_custom.cc
  computation_order/policy_dummy.cc
  computation_order/policy_gt.cc
  custom_onnx_ops.cc
  dtype.cc
  dtype_inference.cc
  evaluator.cc
  file_cache.cc
  ${CMAKE_CURRENT_BINARY_DIR}/flags.cc
  flops.cc
  fusion.cc
  fusion_dldt.cc
  fusion_elementwise.cc
  fusion_ngraph.cc
  fusion_snpe.cc
  fusion_tensorrt.cc
  fusion_tvm.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.cc
  gradient.cc
  gradient_ops.cc
  gradient_with_order.cc
  graph.cc
  graph_builder.cc
  memory_simulator.cc
  merge.cc
  model.cc
  murmur_hash3.cc
  node.cc
  nvrtc_builder.cc
  onnx.cc
  passes.cc
  quantize.cc
  scheduler.cc
  shape_evaluator.cc
  simplifier.cc
  subgraph_canonicalizer.cc
  tensor.cc
  topology.cc
  tvm/compiler.cc
  type.cc
  util.cc
  value.cc
  chxvm/chxvm_value.cc
  chxvm/emitter.cc
  chxvm/simple_node_emitter.cc
  chxvm/value_id_manager.cc
  )
add_dependencies(
  chainer_compiler_compiler
  runtime_chxvm_pb_h compiler_chxvm_codegen_h gen_node_base_h compiler_flags_h
  gen_onnx_proto gen_onnx_operators_proto
  )
set_hidden_(chainer_compiler_compiler)

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(chainer_compiler_compiler_test
  code_emitter_test.cc
  custom_onnx_ops_test.cc
  dtype_inference_test.cc
  evaluator_test.cc
  file_cache_test.cc
  flops_test.cc
  fusion_test.cc
  gradient_test.cc
  merge_test.cc
  model_test.cc
  scheduler_test.cc
  shape_evaluator_test.cc
  simplifier_test.cc
  tensor_test.cc
  topology_test.cc
  chxvm/emitter_test.cc
  )
add_dependencies(
  chainer_compiler_compiler_test
  runtime_chxvm_pb_h compiler_chxvm_codegen_h gen_node_base_h compiler_flags_h
  gen_onnx_proto
  )
target_link_libraries(chainer_compiler_compiler_test
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainer_compiler_configs
  ${CHAINER_COMPILER_DEPENDENCY_LIBRARIES}
  gtest
  gtest_main
  )

add_test(
  NAME chainer_compiler_compiler_test
  COMMAND chainer_compiler_compiler_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
