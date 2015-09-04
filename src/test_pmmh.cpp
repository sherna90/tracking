#include "../include/test_pmmh.hpp"
#include "../include/pmmh.hpp"
#include "../include/utils.hpp"

#include <iostream>

using namespace std;
using namespace cv;

TestPMMH::TestPMMH(ImageGenerator * _img_gen, int _num_particles, int _fixed_lag, int _num_mcmc){
  num_particles = _num_particles;
  fixed_lag = _fixed_lag;
  num_mcmc = _num_mcmc;
  imageGenerator = _img_gen;
  num_frames = imageGenerator->getDatasetSize();
  reinit_rate = 0.0;
}

void TestPMMH::run(){
  pmmh filter(num_particles, fixed_lag, num_mcmc);
  Rect initialization;
  initialization = stringToRect(imageGenerator->getRegion());
  string image_path = imageGenerator->getFrame();
  current_frame = imread(image_path);
  filter.initialize(current_frame, initialization);

  time(&start);
  while(!imageGenerator->isEnded()){
    ground_truth = stringToRect(imageGenerator->getRegion());
    string image_path = imageGenerator->getFrame();
    if (image_path.empty()) break;
    current_frame = imread(image_path);
    if(!filter.is_initialized()){
        filter.initialize(current_frame,ground_truth);
    }
    else if(filter.is_initialized()){
        filter.update(current_frame);
    }
    Rect estimate = filter.estimate(current_frame,true);
    double r1 = performance.calc(ground_truth, estimate);
    if(r1<0.1) {
        filter.reinitialize();
        reinit_rate+=1.0;
    }
  }
  time(&end);
  double sec = difftime (end, start);
  cout  << performance.get_avg_precision()/num_frames;
  cout << "," << performance.get_avg_recall()/num_frames ;
  cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
}
