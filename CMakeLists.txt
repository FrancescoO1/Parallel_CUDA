cmake_minimum_required(VERSION 3.10)

project(KernelImageProcessing)

# Standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flag di ottimizzazione
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Release di default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# File sorgenti
set(SOURCES
        main.cpp
        Image.cpp
        ConvolutionProcessor.cpp
        PerformanceMonitor.cpp
        ImageProcessingManager.cpp
)

# Crea eseguibile
add_executable(${PROJECT_NAME} ${SOURCES})