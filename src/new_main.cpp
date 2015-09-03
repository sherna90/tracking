#include <iostream>
#include <opencv2/core/core.hpp>

#include "../include/test_pmmh.hpp"
#include "../include/image_generator.hpp"
#include "../include/particle_filter.hpp"
#include "../include/utils.hpp"

#define VOT_RECTANGLE
#include "vot.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
  //VOT vot;
  int num_particles = 300;
  int fixed_lag = 3;
  int num_mcmc = 3;

  string firstFrameFilename = "/home/bruno/src/tracking/vot2014/ball/00000001.jpg";
  string groundTruthFile = "/home/bruno/src/tracking/vot2014/ball/groundtruth.txt";
  ImageGenerator imageGenerator(firstFrameFilename, groundTruthFile);

  /*
  while(!imageGenerator.isEnded()){
    cout << imageGenerator.getFrame() << endl;
    cout << imageGenerator.getRegion() << endl;
  }*/

  AlgorithmTest * test;
  // Select algorithm to run
  if(argc != 3) {
      cerr <<"Wrong number of arguments." << endl;
      cerr <<"Usage: program_name --algorithm algorithm_name" << endl;
      cerr <<"Algorithm can be: \"PMMH\"" << endl;
      cerr << "Exiting." << endl;
      return -1;
  }
  string alg_name;

  if(strcmp(argv[1], "--algorithm") == 0){
    alg_name = argv[2];
    cout << "Algorithm selected: " << alg_name << endl;
  }else{
    cerr << "Algorithm not selected." << endl;
    cerr << "Exiting." << endl;
    return -1;
  }

  // Initialize Testing Object
  if(strcmp(alg_name.c_str(), "PMMH") == 0){
    TestPMMH test_pmmh(&imageGenerator, num_particles, fixed_lag, num_mcmc);
    test = &test_pmmh;
    test_pmmh.run();
  }else{
    cerr << "Algorithm name is not valid." << endl;
    return -1;
  }
  // Execute tests on the selected algorithm
  //test->run();
}
