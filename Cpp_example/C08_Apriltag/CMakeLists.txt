cmake_minimum_required(VERSION 3.1)
project(opencv_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置交叉编译工具链
set(PROJECT_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include("${PROJECT_ROOT_PATH}/toolchains/arm-rockchip830-linux-uclibcgnueabihf.toolchain.cmake")

# 配置第三方库路径
set(OpenCV_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/opencv-mobile-4.10.0-lockzhiner-vision-module")
set(OpenCV_DIR "${OpenCV_ROOT_PATH}/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(LockzhinerVisionModule_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk")
set(LockzhinerVisionModule_DIR "${LockzhinerVisionModule_ROOT_PATH}/lib/cmake/lockzhiner_vision_module")
find_package(LockzhinerVisionModule REQUIRED)

# 查找apriltag库

set(APRILTAG_LIB_DIR "${PROJECT_ROOT_PATH}/third_party/apriltag-with-pose-estimation-master/build/lib")
find_library(APRILTAG_LIB NAMES apriltag PATHS ${APRILTAG_LIB_DIR} REQUIRED)

# include_directories("${PROJECT_ROOT_PATH}/third_party/eigen-master/")
include_directories("${PROJECT_ROOT_PATH}/third_party/apriltag-with-pose-estimation-master")

# 定义 Eigen 路径
set(Eigen_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk/third_party/eigen")
set(Eigen_DIR "${Eigen_ROOT_PATH}/include/eigen3/")

# 添加头文件搜索路径
include_directories(
    ${Eigen_DIR}
    ${LOCKZHINER_VISION_MODULE_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)


# 创建可执行文件
add_executable(Test_Apriltag Apriltag.cc)
# 告诉 Eigen 使用最大 8 字节对齐（armv7l 平台限制）
target_compile_definitions(Test_Apriltag PRIVATE EIGEN_MAX_ALIGN_BYTES=8)
add_definitions(-DEIGEN_MAX_ALIGN_BYTES=8)

# 消除 Eigen 的 alignment 警告
target_compile_definitions(Test_Apriltag PRIVATE EIGEN_MAX_ALIGN_BYTES=8)

# 可选：关闭 attributes 警告
target_compile_options(Test_Apriltag PRIVATE -Wno-attributes)



# 链接库文件
target_link_libraries(Test_Apriltag
    ${APRILTAG_LIB}
    ${OpenCV_LIBS}
    ${LOCKZHINER_VISION_MODULE_LIBRARIES}
)

# 输出目录设置
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)