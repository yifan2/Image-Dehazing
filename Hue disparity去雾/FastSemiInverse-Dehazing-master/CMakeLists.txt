cmake_minimum_required (VERSION 2.8)
project (FastSemiInverse)

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
    ${OPENCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")#-O3

#set(CMAKE_C_FLAGS_RELEASE "-O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP")
#set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP")

add_executable(main
    src/main.cpp
)
target_link_libraries(main ${OpenCV_LIBS})
