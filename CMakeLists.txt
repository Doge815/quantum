cmake_minimum_required(VERSION 3.28)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(coin_flip C)
project(bell_state C)
project(deutsch_jozsa C)
project(main C)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_C_COMPILER_VERSION VERSION_LESS 13.2)
        message(WARNING "Old GCC version detected. Only GCC 13.2 has been tested.\nDetected compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    endif()
else()
    message(WARNING "Unsupported compiler detected. Only GCC has been tested.\nDetected compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
endif()


find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -march=native -O3 -mavx2 -fopenmp -lm -Werror -Wall -Wpedantic -Wno-unknown-pragmas")
set(CMAKE_C_STANDARD 23)

file(GLOB simulation_SRC CONFIGURE_DEPENDS src/Simulation/*.c)

add_executable(coin_flip src/examples/coin_flip.c  ${simulation_SRC})
add_executable(bell_state src/examples/bell_state.c ${simulation_SRC})
add_executable(deutsch_jozsa src/examples/deutsch_jozsa.c ${simulation_SRC})
add_executable(main src/main.c ${simulation_SRC})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(coin_flip PUBLIC OpenMP::OpenMP_C)
    target_link_libraries(bell_state PUBLIC OpenMP::OpenMP_C)
    target_link_libraries(deutsch_jozsa PUBLIC OpenMP::OpenMP_C)
endif()
