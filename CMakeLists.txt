cmake_minimum_required(VERSION 2.8.9)
project(CSubNet)
enable_language(C)
set(CMAKE_C_STANDARD 11)

include_directories(src)
file(GLOB SOURCES "src/*.c")

add_executable(CSubNet ${SOURCES})

#link math library in linux
IF( CMAKE_COMPILER_IS_GNUCC)
	target_link_libraries(CSubNet m)
ENDIF( CMAKE_COMPILER_IS_GNUCC)
