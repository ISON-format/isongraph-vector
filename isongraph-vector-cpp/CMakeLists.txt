cmake_minimum_required(VERSION 3.14)
# C as well as CXX: the optional SQLite backend compiles the SQLite and
# sqlite-vec amalgamations, which are C.
project(isongraph-vector-cpp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(isongraph_vector INTERFACE)
target_include_directories(isongraph_vector INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # ison_graph.hpp and ison_parser.hpp, vendored under
    # vendor/ison-graph-cpp/. There is no package registry for C++, so unlike
    # the other ports these are copies rather than a dependency declaration.
    # Both were taken from the canonical checkouts - ison-graph-cpp 1.4.0 and
    # ison-cpp 1.2.0 - replacing an earlier copy that turned out to be a stale
    # backup at 1.0.0. ISONGraph refuses to compile against an ison-cpp older
    # than 1.2.0 (earlier releases lose floats in transit), which is why the
    # parser header is vendored alongside it.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/ison-graph-cpp>
    $<INSTALL_INTERFACE:include>
)

# Install rules. The interface above already declares an INSTALL_INTERFACE
# include path; without these nothing was ever placed there, so a consumer
# could not find_package() this library.
include(GNUInstallDirs)
install(TARGETS isongraph_vector EXPORT isongraph-vector-targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY vendor/ison-graph-cpp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT isongraph-vector-targets
    FILE isongraph-vector-config.cmake
    NAMESPACE isongraph::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/isongraph-vector
)

# Optional SQLite-backed store. Off by default so the library stays
# header-only and dependency-free; when on, the SQLite and sqlite-vec
# amalgamations are fetched and compiled in, so no system SQLite is needed.
option(ISONGRAPH_VECTOR_SQLITE "Build the SQLite-backed store and its tests" OFF)

if(ISONGRAPH_VECTOR_SQLITE)
    include(FetchContent)

    FetchContent_Declare(sqlite3_amalgamation
        URL https://sqlite.org/2024/sqlite-amalgamation-3450300.zip
        URL_HASH SHA256=ea170e73e447703e8359308ca2e4366a3ae0c4304a8665896f068c736781c651
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_Declare(sqlite_vec_amalgamation
        URL https://github.com/asg017/sqlite-vec/releases/download/v0.1.9/sqlite-vec-0.1.9-amalgamation.tar.gz
        URL_HASH SHA256=3acd67cb4aff080c7050926fd3cf8227905fe5b7ee3829d8ee5024ab1283cf61
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(sqlite3_amalgamation sqlite_vec_amalgamation)

    add_library(isongraph_vector_sqlite_backend STATIC
        ${sqlite3_amalgamation_SOURCE_DIR}/sqlite3.c
        ${sqlite_vec_amalgamation_SOURCE_DIR}/sqlite-vec.c
    )
    # BUILD_INTERFACE only: these live in the build tree, and the fetched
    # backend is a convenience for building and testing here rather than
    # something to install. A consumer who wants the SQLite store either uses
    # add_subdirectory with this option on, or links their own SQLite and
    # includes isongraph_vector_sqlite.hpp directly.
    target_include_directories(isongraph_vector_sqlite_backend PUBLIC
        $<BUILD_INTERFACE:${sqlite3_amalgamation_SOURCE_DIR}>
        $<BUILD_INTERFACE:${sqlite_vec_amalgamation_SOURCE_DIR}>
    )
    # sqlite-vec registers through sqlite3_auto_extension, which needs the
    # loadable-extension shim disabled when the code is compiled in directly.
    target_compile_definitions(isongraph_vector_sqlite_backend PUBLIC
        SQLITE_CORE=1
        SQLITE_ENABLE_MATH_FUNCTIONS=1
    )

    add_library(isongraph_vector_sqlite INTERFACE)
    target_link_libraries(isongraph_vector_sqlite INTERFACE
        isongraph_vector isongraph_vector_sqlite_backend)

endif()

# Tests. Skipped when this project is consumed via add_subdirectory.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
enable_testing()
add_executable(test_isongraph_vector tests/test_isongraph_vector.cpp)
target_link_libraries(test_isongraph_vector PRIVATE isongraph_vector)
add_test(NAME test_isongraph_vector COMMAND test_isongraph_vector)

if(ISONGRAPH_VECTOR_SQLITE)
    add_executable(test_isongraph_vector_sqlite tests/test_isongraph_vector_sqlite.cpp)
    target_link_libraries(test_isongraph_vector_sqlite PRIVATE isongraph_vector_sqlite)
    add_test(NAME test_isongraph_vector_sqlite COMMAND test_isongraph_vector_sqlite)
endif()
endif()
