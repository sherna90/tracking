#include <iostream>
#include <opencv2/core/core.hpp>
#include <string>

#include "../include/test_pmmh.hpp"
#include "../include/test_particle_filter.hpp"

#include "../include/image_generator.hpp"
#include "../include/particle_filter.hpp"
#include "../include/utils.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
  int num_particles = 300;
  int fixed_lag = 3;
  int num_mcmc = 3;

  //AlgorithmTest * test;
  // Select algorithm to run
  if(argc != 5) {
      cerr <<"Wrong number of arguments." << endl;
      cerr <<"Usage: program_name --algorithm algorithm_name --dataset" << endl;
      cerr <<"Algorithm can be: \"PMMH\", \"PARTICLE_FILTER\"" << endl;
      cerr << "Exiting." << endl;
      return -1;
  }
  string dataset(argv[4]);
  string firstFrameFilename = dataset+"/00000001.jpg";
  string groundTruthFile = dataset+"/groundtruth.txt";
  ImageGenerator imageGenerator(firstFrameFilename, groundTruthFile);

  string alg_name;
  if(strcmp(argv[1], "--algorithm") == 0){
    alg_name = argv[2];
    //cout << "Algorithm selected: " << alg_name << endl;
  }else{
    cerr << "Algorithm not selected." << endl;
    cerr << "Exiting." << endl;
    return -1;
  }

  // Initialize Testing Object
  if(strcmp(alg_name.c_str(), "PMMH") == 0){
    TestPMMH test_pmmh(&imageGenerator, num_particles, fixed_lag, num_mcmc);
    //test = &test_pmmh;
    test_pmmh.run();
  }else if(strcmp(alg_name.c_str(), "PARTICLE_FILTER") == 0){
    TestParticleFilter test_particle_filter(&imageGenerator, num_particles);
    test_particle_filter.run();
  }else{
    cerr << "Algorithm name is not valid." << endl;
    return -1;
  }
  // Execute tests on the selected algorithm
  //test->run();
}
