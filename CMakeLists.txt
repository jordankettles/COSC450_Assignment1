CMAKE_MINIMUM_REQUIRED(VERSION "3.12")

set(CMAKE_CXX_STANDARD 14)

project(Mosaicing)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(mosaicing mosaicing.cpp Timer.h Timer.cpp)

target_link_libraries(mosaicing ${OpenCV_LIBS}
)

message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBS: " ${OpenCV_LIBS})
