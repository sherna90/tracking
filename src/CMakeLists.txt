cmake_minimum_required(VERSION 2.8)
project( Examples )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( tracker main.cpp particle_filter.cpp hist.cpp)
#add_executable( tracker tracker.cpp)
target_link_libraries( tracker ${OpenCV_LIBS} )
