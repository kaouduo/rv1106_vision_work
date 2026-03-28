cmake_minimum_required(VERSION 3.10)

project(eigen_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义项目根目录路径
set(PROJECT_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")
message("PROJECT_ROOT_PATH = " ${PROJECT_ROOT_PATH})

include("${PROJECT_ROOT_PATH}/toolchains/arm-rockchip830-linux-uclibcgnueabihf.toolchain.cmake")

# 定义 OpenCV SDK 路径
set(OpenCV_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/opencv-mobile-4.10.0-lockzhiner-vision-module")
set(OpenCV_DIR "${OpenCV_ROOT_PATH}/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
set(OPENCV_LIBRARIES "${OpenCV_LIBS}")

# 定义 LockzhinerVisionModule SDK 路径
set(LockzhinerVisionModule_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk")
set(LockzhinerVisionModule_DIR "${LockzhinerVisionModule_ROOT_PATH}/lib/cmake/lockzhiner_vision_module")
find_package(LockzhinerVisionModule REQUIRED)


# 定义 Eigen 路径
set(Eigen_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk/third_party/eigen")
set(Eigen_DIR "${Eigen_ROOT_PATH}/include/eigen3/")

message("Eigen_DIR = " ${Eigen_DIR})



add_executable(eigen_test eigentest.cc)
target_include_directories(eigen_test PRIVATE ${LOCKZHINER_VISION_MODULE_INCLUDE_DIRS} ${Eigen_DIR})
target_link_libraries(eigen_test PRIVATE ${OPENCV_LIBRARIES} ${LOCKZHINER_VISION_MODULE_LIBRARIES} )
# 告诉 Eigen 使用最大 8 字节对齐（armv7l 平台限制）
target_compile_definitions(eigen_test PRIVATE EIGEN_MAX_ALIGN_BYTES=8)
# 如果你不需要 Eigen 的 SIMD 加速（在嵌入式平台上有时反而会拖慢性能），具体怎样还没有测试，请自行测试
target_compile_definitions(eigen_test PRIVATE EIGEN_DONT_VECTORIZE)


install(
    TARGETS eigen_test
    RUNTIME DESTINATION .  
)