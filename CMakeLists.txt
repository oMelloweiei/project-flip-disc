# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(https_server)

# Set the C++ standard version to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenSSL package
find_package(OpenSSL REQUIRED)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Specify the include directories for OpenSSL and OpenCV
include_directories(
    /opt/homebrew/Cellar/openssl@3/3.4.0/include
    /opt/homebrew/Cellar/opencv/4.10.0_15/include/opencv4
)

# Specify the library directories for OpenSSL and OpenCV
link_directories(
    /opt/homebrew/Cellar/openssl@3/3.4.0/lib
    /opt/homebrew/Cellar/opencv/4.10.0_15/lib
)

# Add the executable target
add_executable(https_server server.cpp)

# Link OpenSSL and OpenCV libraries to your project
target_link_libraries(https_server
    OpenSSL::SSL
    OpenSSL::Crypto
    ${OpenCV_LIBS}  # Link OpenCV libraries
)

# Optionally, you can specify additional flags or settings if needed
# Example for linking with pthread (in case it's required):
# target_link_libraries(https_server pthread)
