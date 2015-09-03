#include "../include/test_pmmh.hpp"
#include "../include/pmmh.hpp"

#include <iostream>

using namespace std;
using namespace cv;

TestPMMH::TestPMMH(ImageGenerator * _img_gen, int _num_particles, int _fixed_lag, int _num_mcmc){
  cout << "Test PMMH initialized." << endl;
  num_particles = _num_particles;
  fixed_lag = _fixed_lag;
  num_mcmc = _num_mcmc;
  imageGenerator = _img_gen;
  cout << stringToRect(imageGenerator->getRegion()) << endl;
}

void TestPMMH::run(){
  cout << "Running PMMH test" << endl;
  pmmh filter(num_particles, fixed_lag, num_mcmc);
  Rect initialization;
  initialization = stringToRect(imageGenerator->getRegion());
  string image_path = imageGenerator->getFrame();
  current_frame = imread(image_path);
  filter.initialize(current_frame, initialization);
  while(!imageGenerator->isEnded()){
    string image_path = imageGenerator->getFrame();
    if (image_path.empty()) break;
    current_frame = imread(image_path);
    filter.update(current_frame);
    Rect estimate = filter.estimate(current_frame,true);
  }
}
