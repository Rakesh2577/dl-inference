cmake_minimum_required(VERSION 3.16)

project(DL_Inference_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(dl_inference main.cpp)

target_include_directories(dl_inference PRIVATE
    /opt/homebrew/include/onnxruntime
)

target_link_directories(dl_inference PRIVATE
    /opt/homebrew/lib
)

target_link_libraries(dl_inference PRIVATE
    onnxruntime
)
