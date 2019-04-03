cmake_minimum_required(VERSION 3.13)
project(pl_ICP)

set(CMAKE_CXX_STANDARD 14)

INCLUDE (FindGSL.cmake)
IF(GSL_FOUND)
    MESSAGE(STATUS "Found GSL:")
    MESSAGE(STATUS "   GSL_LIBRARIES = ${GSL_LIBRARIES}")
    MESSAGE(STATUS "   GSL_INCLUDE_DIR = ${GSL_INCLUDE_DIR}")
    MESSAGE(STATUS "   CMAKE_GSL_CXX_FLAGS = ${CMAKE_GSL_CXX_FLAGS}")
    MESSAGE(STATUS "   GSL_LINK_DIRECTORIES = ${GSL_LINK_DIRECTORIES}")

    INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIR})
    LINK_DIRECTORIES(${GSL_LINK_DIRECTORIES})
ELSE(GSL_FOUND)
    MESSAGE(ERROR "GSL not found.")
ENDIF(GSL_FOUND)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pl_ICP main.cpp gpc.cpp gpc.h)
target_link_libraries(pl_ICP ${GSL_LIBRARIES} ${CERES_LIBRARIES} ${Sophus_LIBRARIES} ${PCL_LIBRARIES})