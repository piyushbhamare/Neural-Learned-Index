cmake_minimum_required(VERSION 3.15)
project(neural_learned_index VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

# Add source directories
add_subdirectory(src)

# Optional: Add tests if gtest is available
find_package(GTest)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile)
endif()
