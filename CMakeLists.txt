cmake_minimum_required(VERSION 3.2)
project(RPCA_CPP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Armadillo)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# library output dir
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(RPCA SHARED
        include/svd_truncated.h
        src/svd_truncated.cpp
        include/RPCA.h
        src/RPCA.cpp
        )
target_link_libraries(RPCA
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        armadillo)

# test demo
add_executable(test_demo src/test_demo.cpp include/svd_truncated.h include/RPCA.h)
target_link_libraries(test_demo
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        armadillo
        ${PROJECT_SOURCE_DIR}/lib/libRPCA.so)
