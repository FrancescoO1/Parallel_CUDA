cmake_minimum_required(VERSION 3.18)
project(Parallel_CUDA_Orlandi_Francesco LANGUAGES CXX CUDA)

# Imposta lo standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Imposta lo standard CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Abilita CUDA
enable_language(CUDA)

# Trova CUDA (metodo moderno)
find_package(CUDAToolkit REQUIRED)

# Mostra informazioni
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# Definisci i file sorgente
set(SOURCES
        Image.cpp
        CudaConvolutionProcessor.cu
        main.cpp
        ConvolutionProcessor.cpp
)

# Crea l'eseguibile
add_executable(${PROJECT_NAME} ${SOURCES})

# Imposta le proprietà CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50;60;70;75;80;86"
)

# Collega le librerie CUDA
target_link_libraries(${PROJECT_NAME}
        CUDA::cudart
        CUDA::cuda_driver
)

# Flag di compilazione per debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-g -O0 -Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-g -G -O0>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
    )
endif()

# Include directory (se necessario)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
