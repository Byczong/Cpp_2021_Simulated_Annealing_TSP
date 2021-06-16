cmake_minimum_required(VERSION 3.19)
project(Simulated_annealing)

set(CMAKE_CXX_STANDARD 17)

set(SFML_ROOT /home/byczong/Documents/Studia/Programowanie_w_cpp/Simulated_annealing/SFML)
set(SFML_DIR "SFML/lib/cmake/SFML")

set(CMAKE_MODULE_PATH "$(CMAKE_CURRENT_LIST_DIR)/cmake_modules")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package(SFML COMPONENTS audio graphics window system)
add_executable(Simulated_annealing main.cpp annealing.cpp annealing.h application.h application.cpp)

if(SFML_FOUND)
    include_directories(${SFML_INCLUDE_DIR})
    target_link_libraries(Simulated_annealing sfml-graphics sfml-audio sfml-window sfml-system)
endif ()