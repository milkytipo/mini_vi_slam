cmake_minimum_required(VERSION 3.8.11)
project(mini_vi_slam)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set(Ceres_LIBS_DIR  ~/3rdlib/ceres-solver-1.14.0/build/lib)
set(Ceres_DIR ~/3rdlib/ceres-solver-1.14.0/build)
message(${Ceres_DIR})

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)



include(CTest)


add_subdirectory(apps)

add_subdirectory(lib)

option(BUILD_TEST "Build tests." ON)
if(BUILD_TEST)
  add_subdirectory(test)
endif()
