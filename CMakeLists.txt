cmake_minimum_required(VERSION 3.15)
project(onnxTest)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)


set(ONNXRUNTIME_ROOT_PATH /Users/zhanghaining/CLionProjects/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime//core/providers/cuda

        )
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/MacOS/Release/libonnxruntime.1.11.0.dylib)

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable(main main.cpp Onnx.h Onnx.cpp UTIL.h UTIL.cpp)
target_link_libraries(main ${ONNXRUNTIME_LIB})
target_link_libraries(main ${OpenCV_LIBS})


