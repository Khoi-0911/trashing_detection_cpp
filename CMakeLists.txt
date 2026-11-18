cmake_minimum_required(VERSION 3.23)
project(trashing_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(TRT_DIR "C:/libs/TensorRT-11.4.0.106")
if(WIN32)
    set(NVINFER nvinfer_11)
else()
    set(NVINFER nvinfer)
endif()

add_executable(trashing_detection trashing_detection.cpp)

find_package(OpenCV REQUIRED)
target_include_directories(trashing_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trashing_detection PRIVATE ${OpenCV_LIBS})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(trashing_detection PRIVATE CUDA::cudart)

target_include_directories(trashing_detection PRIVATE ${TRT_DIR}/include)
target_link_directories(trashing_detection PRIVATE ${TRT_DIR}/lib)
target_link_libraries(trashing_detection PRIVATE ${NVINFER})