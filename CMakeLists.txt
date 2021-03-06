cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED)

OPTION (MAKE_HMC "Make HMC" OFF)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/detector/" )
include_directories( "src/likelihood/" )
include_directories( "src/models/" )
include_directories( "src/features/" )
include_directories( "src/utils/" )
include_directories( "src/DPP/" )
include_directories( "/usr/include/eigen3/" )

add_subdirectory("src/libs/piotr_fhog")



set(COMMON_SOURCES
	src/utils/utils.cpp 
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/CPU_logistic_regression.cpp  
	src/likelihood/multivariate_gaussian.cpp  
	src/detector/hog_detector.cpp 
	src/likelihood/CPU_logistic_regression.cpp 
	src/detector/CPU_LR_hog_detector.cpp 
	src/DPP/dpp.cpp 
	src/features/hist.cpp
    )



add_executable( bernoulli_pf src/test_bernoulli_particle_filter.cpp src/models/bernoulli_particle_filter.cpp  ${COMMON_SOURCES}) 
target_link_libraries( bernoulli_pf ${OpenCV_LIBS} fhog )


add_executable( particle_filter src/test_particle_filter.cpp src/models/particle_filter.cpp  ${COMMON_SOURCES} ) 
target_link_libraries( particle_filter ${OpenCV_LIBS} fhog )
