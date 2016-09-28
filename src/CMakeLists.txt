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

add_executable( pmmh test_pmmh.cpp pmmh.cpp particle_filter.cpp  utils.cpp gaussian.cpp  image_generator.cpp haar.cpp gaussian_naivebayes.cpp logistic_regression.cpp multivariate_gaussian.cpp)
target_link_libraries( pmmh ${OpenCV_LIBS}  )

add_executable( tracker test_particle_filter.cpp particle_filter.cpp  utils.cpp gaussian.cpp image_generator.cpp haar.cpp logistic_regression.cpp gaussian_naivebayes.cpp multivariate_gaussian.cpp) 
target_link_libraries( tracker ${OpenCV_LIBS} )

add_executable( smc_squared test_smcsquared.cpp  smc_squared.cpp pmmh.cpp particle_filter.cpp utils.cpp gaussian.cpp image_generator.cpp haar.cpp multivariate_gaussian.cpp gaussian_naivebayes.cpp logistic_regression.cpp multivariate_gaussian.cpp) 
target_link_libraries( smc_squared ${OpenCV_LIBS}  )

#FILE(GLOB_RECURSE kcf_sourcefiles "kcf/*.cpp")
#add_executable( kcf test_kcf.cpp ${kcf_sourcefiles} utils.cpp image_generator.cpp)
#target_link_libraries( kcf ${OpenCV_LIBS} )

#add_executable( ct test_ct.cpp ct/CompressiveTracker.cpp utils.cpp image_generator.cpp)
#target_link_libraries( ct ${OpenCV_LIBS} )
