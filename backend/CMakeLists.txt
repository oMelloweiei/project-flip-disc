# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and language
project(https_server CXX)

# Set the C++ standard version to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)

# Print status messages for debugging
message(STATUS "OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

# Add the executable target
add_executable(https_server main.cpp)

# Set include directories using modern CMake
target_include_directories(https_server
    PRIVATE
        ${OPENSSL_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

# Link libraries using modern CMake
target_link_libraries(https_server
    PRIVATE
        OpenSSL::SSL
        OpenSSL::Crypto
        ${OpenCV_LIBS}
)

# Optional: Add compile options
target_compile_options(https_server
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
)

# Optional: Enable threading support
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(https_server PRIVATE Threads::Threads)
endif()