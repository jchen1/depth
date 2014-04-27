cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
file( GLOB depth_SRC
      "src/*.h"
      "src/*.cpp"
)
add_executable( depth ${depth_SRC} )
add_definitions( -std=c++11 )
target_link_libraries( depth ${OpenCV_LIBS} )