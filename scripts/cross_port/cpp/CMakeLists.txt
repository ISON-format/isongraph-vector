# C++ half of the graph-file cross-port check (see ../graph_files.py).
#
# A standalone project that pulls the port in by relative path. The port is
# header-only, so nothing is built in its directory; the orchestrator configures
# this project with a build directory outside the repository.
cmake_minimum_required(VERSION 3.14)
project(cross_port_graph LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../isongraph-vector-cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/isongraph-vector-cpp EXCLUDE_FROM_ALL)

add_executable(cross_port_graph main.cpp)
target_link_libraries(cross_port_graph PRIVATE isongraph_vector)
if(MSVC)
    # The fixture has non-ASCII string literals; read the source as UTF-8.
    target_compile_options(cross_port_graph PRIVATE /utf-8)
endif()
