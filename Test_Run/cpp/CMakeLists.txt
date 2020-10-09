
cmake_minimum_required(VERSION 3.1.0)
project(rPPG-v1)

set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS " -Ofast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")

add_subdirectory(../dlib dlib_build)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

set(THREAD_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES ./src/opencv_ex.cpp)

add_executable(rPPG-v1 ${SOURCE_FILES})
target_link_libraries(rPPG-v1 Threads::Threads dlib::dlib ${realsense2_LIBRARY} ${OpenCV_LIBS} )
