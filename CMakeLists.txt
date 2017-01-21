cmake_minimum_required(VERSION 3.0.0)
project(CSubNet)
enable_language(C)
set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)

include_directories(src)
file(GLOB SOURCES "src/*.c")

add_executable(CSubNet ${SOURCES})

if(CMAKE_COMPILER_IS_GNUCC)
	#GCC Options
	#link math library
	target_link_libraries(CSubNet m)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -march=native -fopenmp")
elseif (MSVC)
	#MSVC Options
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast /arch:SSE2 /openmp")
endif()
