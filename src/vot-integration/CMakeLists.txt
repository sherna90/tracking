PROJECT(vot_integration)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -std=c++11 -fopenmp")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED)
find_path(FFTW_INCLUDE_DIR fftw3.h  ${FFTW_INCLUDE_DIRS})
find_library(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "../likelihood/" )
include_directories( "../models/" )
include_directories( "../features/" )
include_directories( "../utils/" )
include_directories( "../DPP/" )
include_directories( "../libs/LBP/" )
include_directories( "/usr/include/eigen3/" )
#include_directories( "/usr/local/include/eigen3/" )

# Try to find TraX header and library ...
FIND_FILE(TRAX_HEADER NAMES trax.h PATHS ../../../native)
IF (EXISTS ${TRAX_HEADER})
    GET_FILENAME_COMPONENT(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
    ADD_DEFINITIONS(-DTRAX) # Added to enable TraX protocol support
    LINK_DIRECTORIES(${TRAX_DIR}) # Add library directory
    INCLUDE_DIRECTORIES(${TRAX_DIR}) # Add includes directory
ENDIF()

ADD_EXECUTABLE(bernoulli_pf  vot_bernoulli_particle_filter.cpp ../models/bernoulli_particle_filter.cpp ../DPP/dpp.cpp ../utils/utils.cpp ../likelihood/gaussian.cpp ../utils/image_generator.cpp ../features/haar.cpp ../features/local_binary_pattern.cpp ../likelihood/logistic_regression.cpp ../likelihood/hmc.cpp ../likelihood/incremental_gaussiannaivebayes.cpp ../likelihood/multivariate_gaussian.cpp ../likelihood/multinomial.cpp  ../libs/LBP/LBP.cpp) # Generate executable for C++ tracker

IF (EXISTS ${TRAX_HEADER})
    TARGET_LINK_LIBRARIES(bernoulli_pf trax ${OpenCV_LIBS} ${FFTW_LIBRARY}) # Link with trax library
ENDIF()
TARGET_LINK_LIBRARIES(bernoulli_pf ${OpenCV_LIBS} ${FFTW_LIBRARY})
