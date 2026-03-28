cmake_minimum_required(VERSION 3.10)

project(jsontest)

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


# 定义 jsoncpp 路径
set(JSONCPP_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk/third_party/jsoncpp")
set(JSONCPP_INCLUDE_DIRS "${JSONCPP_ROOT_PATH}/include/")
set(JSONCPP_LIBRARIES "${JSONCPP_ROOT_PATH}/lib/libjsoncpp.a")

message("JSONCPP_INCLUDE_DIRS = " ${JSONCPP_INCLUDE_DIRS})

add_executable(jsontest jsontest.cc)
target_include_directories(jsontest PRIVATE ${LOCKZHINER_VISION_MODULE_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(jsontest PRIVATE ${OPENCV_LIBRARIES} ${LOCKZHINER_VISION_MODULE_LIBRARIES} ${JSONCPP_LIBRARIES})

install(
    TARGETS jsontest
    RUNTIME DESTINATION .  
)