cmake_minimum_required(VERSION 3.12)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(xiaodu_hi_infer CXX C)

# Options for Paddle
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT    "Compile demo with TensorRT."   OFF)

# Options for infer_v*.cpp compile mode
option(USE_SERVER_MODE   "Compile code under http or grpc server mode." ON)
option(WITH_LOCAL_CLIENT "Compile code with local client to access camera device." OFF)
option(TESTCASE_ONLY     "Compile code with only unit testcases, useful for dev." OFF)
option(LOCAL_INFER       "Compile code to load video file for inference. When USEER_SERVER_MODE is ON, this option is ignored." ON)
option(ASYNC_INFER       "Compile code with async inference pipeline. When USE_SERVER_MODE is OFF, this option is ignored." ON)

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED INFER_NAME)
  message(FATAL_ERROR "please set INFER_NAME with -DINFER_NAME=infer_name")
endif()

# Find OpenCV, need to set OpenCV_DIR variable when run `cmake`
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Find Eigen3
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(INFER_NAMES_USING_gRPC "infer_v3" "eval_v3" "infer_r2plus1d" "eval_r2plus1d")
if(USE_SERVER_MODE AND (${INFER_NAME} IN_LIST INFER_NAMES_USING_gRPC))
  set(USE_gRPC ON)
else()
  set(USE_gRPC OFF)
endif()

# Find gRPC and its dependencies, then compile proto file
macro(find_gRPC_and_deps Proto_FILE)
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${protobuf_VERSION}")

  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_REFLECTION gRPC::grpc++_reflection)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  set(_GRPC_GRPCPP gRPC::grpc++)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

  # Proto file
  get_filename_component(proto "${Proto_FILE}" ABSOLUTE)
  get_filename_component(proto_path "${proto}" PATH)
  message(STATUS "Proto: ${proto}")
  message(STATUS "Proto path: ${proto_path}")

  string(REPLACE ".proto" "" prefix_name "${Proto_FILE}")
  message(STATUS "Prefix name: ${prefix_name}")

  # Generated sources
  set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${prefix_name}.pb.cc")
  set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${prefix_name}.pb.h")
  set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${prefix_name}.grpc.pb.cc")
  set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${prefix_name}.grpc.pb.h")
  add_custom_command(
      OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${proto}"
      DEPENDS "${proto}")

  # Include generated *.pb.h files
  include_directories("${CMAKE_CURRENT_BINARY_DIR}")
endmacro()

# Compile proto for infer_v3.cpp server mode
if(USE_gRPC AND (${INFER_NAME} STREQUAL "infer_v3"))
  find_package(Threads REQUIRED)
  find_gRPC_and_deps("proactive_greeting.proto")
endif()

# Compile proto for infer_r2plus1d.cpp server mode
if(USE_gRPC AND (${INFER_NAME} STREQUAL "infer_r2plus1d"))
  find_package(Threads REQUIRED)
  find_gRPC_and_deps("proactive_greeting.proto")
endif()

# Compile proto for eval_v3.cpp server mode
if(USE_gRPC AND (${INFER_NAME} STREQUAL "eval_v3"))
  find_package(Threads REQUIRED)
  find_gRPC_and_deps("eval_server.proto")
endif()

# Compile proto for eval_r2plus1d.cpp server mode
if(USE_gRPC AND (${INFER_NAME} STREQUAL "eval_r2plus1d"))
  find_package(Threads REQUIRED)
  find_gRPC_and_deps("eval_server.proto")
endif()

# Find Boost
find_package(Boost 1.53.0 COMPONENTS system thread filesystem)

if(USE_SERVER_MODE)
  if(NOT USE_gRPC)
    # For http server
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/simpleweb")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/base64")
  endif()

  add_compile_definitions(SERVER_MODE)
endif()

if(WITH_LOCAL_CLIENT)
  add_compile_definitions(LOCAL_CLIENT)
endif()

if(TESTCASE_ONLY)
  add_compile_definitions(TESTCASE_ONLY)
endif()

if(LOCAL_INFER)
  add_compile_definitions(LOCAL_INFER)
endif()

if(ASYNC_INFER)
  add_compile_definitions(ASYNC_INFER)
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wextra")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGS})

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")

if (USE_TENSORRT AND WITH_GPU)
      include_directories("${PADDLE_LIB}/third_party/install/tensorrt/include")
      link_directories("${PADDLE_LIB}/third_party/install/tensorrt/lib")
endif()

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

# link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

if(USE_gRPC)
  add_executable(${INFER_NAME} ${INFER_NAME}.cpp ${proto_srcs} ${grpc_srcs})
else()
  add_executable(${INFER_NAME} ${INFER_NAME}.cpp)
endif()

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
               ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

#set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(EXTERNAL_LIB "-ldl -lpthread")

set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    glog gflags z xxhash
    ${EXTERNAL_LIB})

if(WITH_GPU)
  if (USE_TENSORRT)
    set(DEPS ${DEPS}
        ${PADDLE_LIB}/third_party/install/tensorrt/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS}
        ${PADDLE_LIB}/third_party/install/tensorrt/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDA_LIB}/libcublas${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX} )

  # By default, enable CUDA accelerated optical flow computation when use GPU
  add_compile_definitions(CUDA_OPT_FLOW)
endif()

if(USE_SERVER_MODE)

  if(USE_gRPC)
    target_link_libraries(${INFER_NAME} ${DEPS} ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
  else()
    target_link_libraries(${INFER_NAME} ${DEPS} ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen)
  endif()

else()
  target_link_libraries(${INFER_NAME} ${DEPS} ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen)
endif()
