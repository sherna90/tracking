cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "/usr/local/include/eigen3/" )

# Try to find TraX header and library ...
FIND_FILE(TRAX_HEADER NAMES trax.h PATHS /Users/felipe/Documents/codes/matlab/vot-toolkit/native)

IF (EXISTS ${TRAX_HEADER})
    GET_FILENAME_COMPONENT(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
    ADD_DEFINITIONS(-DTRAX) # Added to enable TraX protocol support
    LINK_DIRECTORIES(${TRAX_DIR}) # Add library directory
    INCLUDE_DIRECTORIES(${TRAX_DIR}) # Add includes directory
ENDIF()

add_executable( particle_filter main.cpp ../particle_filter.cpp ../utils.cpp ../gaussian.cpp ../image_generator.cpp ../haar.cpp)

IF (EXISTS ${TRAX_HEADER})
    target_link_libraries(particle_filter ${TRAX_DIR} -DTRAX -ltraxstatic ${OpenCV_LIBS})
    #target_link_libraries(particle_filter trax ${OpenCV_LIBS})
ENDIF()
