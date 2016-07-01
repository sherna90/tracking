cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_C_COMPILER             	"icc")
SET (CMAKE_CXX_COMPILER             "icpc")
SET (CMAKE_CXX_FLAGS                "-Wall -g -w -O3 -std=c++11")
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( "/usr/include/eigen3/" )
#include_directories( "/usr/local/include/eigen3/" )
include_directories("/opt/intel/compilers_and_libraries/linux/mkl/include")
link_directories("/opt/intel/compilers_and_libraries/linux/mkl/lib/")

include_directories( "LBP/" )
find_path(FFTW_INCLUDE_DIR fftw3.h  ${FFTW_INCLUDE_DIRS})
find_library(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})

add_library(LBP SHARED LBP/LBP.cpp )
target_link_libraries(LBP ${OpenCV_LIBS} ${FFTW_LIBRARY}  m)

add_executable( pmmh test_pmmh.cpp pmmh.cpp particle_filter.cpp hist.cpp hog.cpp dirichlet.cpp utils.cpp gaussian.cpp multinomial.cpp poisson.cpp  image_generator.cpp haar.cpp)
target_link_libraries( pmmh ${OpenCV_LIBS} -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -lm LBP ${FFTW_LIBRARY} )

add_executable( tracker test_particle_filter.cpp particle_filter.cpp  hist.cpp hog.cpp dirichlet.cpp utils.cpp gaussian.cpp multinomial.cpp poisson.cpp image_generator.cpp haar.cpp) 
target_link_libraries( tracker ${OpenCV_LIBS} -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -lm LBP ${FFTW_LIBRARY} )
