cmake_minimum_required(VERSION 2.8.9)
project(CSubNet)
enable_language(C)
set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE RelWithDebInfo )

include_directories(src)
file(GLOB SOURCES "src/*.c")

add_executable(CSubNet ${SOURCES})

IF(CMAKE_COMPILER_IS_GNUCC)
	#GCC Options
	#link math library
	target_link_libraries(CSubNet m)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	#MSVC Options
ENDIF()
