cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")
find_package( OpenCV REQUIRED)
#find_package(CUDA QUIET REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/detector/" )
include_directories( "src/likelihood/" )
include_directories( "src/models/" )
include_directories( "src/features/" )
include_directories( "src/utils/" )
include_directories( "src/DPP/" )
include_directories( "/usr/include/eigen3/" )
#include_directories( "/usr/local/cuda-8.0/include/" )
#include_directories( "/usr/local/include/eigen3/" )

set(CF_PIOTR_DIR "src/libs/piotr")
include_directories(${CF_PIOTR_DIR} ${CF_PIOTR_DIR}/src)

set(CF_PIOTR_SOURCES
	${CF_PIOTR_DIR}/shift.hpp
	${CF_PIOTR_DIR}/math_spectrums.hpp
	${CF_PIOTR_DIR}/cv_ext.hpp
	${CF_PIOTR_DIR}/mat_consts.hpp
	${CF_PIOTR_DIR}/math_helper.hpp
	${CF_PIOTR_DIR}/feature_channels.hpp
    ${CF_PIOTR_DIR}/gradientMex.hpp
    ${CF_PIOTR_DIR}/src/gradientMex.cpp
    ${CF_PIOTR_DIR}/src/sse.hpp
    ${CF_PIOTR_DIR}/src/wrappers.hpp)

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
    ${CF_PIOTR_SOURCES})

set(HMC_SOURCES
	src/likelihood/hmc.cpp  
	src/likelihood/CPU_hmc.cpp  
	src/detector/CPU_HMC_hog_detector.cpp 
   )

add_executable( bernoulli_pf src/test_bernoulli_particle_filter.cpp src/models/bernoulli_particle_filter.cpp  ${COMMON_SOURCES}) 
target_link_libraries( bernoulli_pf ${OpenCV_LIBS} )

add_executable( bernoulli_pf_HMC src/test_bernoulli_particle_filter_HMC.cpp src/models/bernoulli_particle_filter_HMC.cpp  ${COMMON_SOURCES} ${HMC_SOURCES}) 
target_link_libraries( bernoulli_pf_HMC ${OpenCV_LIBS} )

add_executable( particle_filter src/test_particle_filter.cpp src/models/particle_filter.cpp  ${COMMON_SOURCES} ) 
target_link_libraries( particle_filter ${OpenCV_LIBS} )