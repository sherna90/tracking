#include <iostream>

#include "../include/algorithm_test.hpp"
#include "../include/test_pmmh.hpp"
#include "../include/particle_filter.hpp"
#include "../include/utils.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
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
    TestPMMH test_pmmh;
    test = &test_pmmh;
  }else{
    cerr << "Algorithm name is not valid." << endl;
    return -1;
  }

  // Execute tests
  test->run();
}
