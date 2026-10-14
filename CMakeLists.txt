cmake_minimum_required(VERSION 3.25)

project(parquet_format_study VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_executable(parquet_generator
    src/main.cpp
)

target_link_libraries(parquet_generator
    PRIVATE
        Arrow::arrow_shared
        Parquet::parquet_shared
)

target_compile_options(parquet_generator PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)

target_compile_definitions(parquet_generator PRIVATE
    PROJECT_VERSION=\"${PROJECT_VERSION}\"
)
