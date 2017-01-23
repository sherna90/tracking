#include "DPP/dpp_tracker.hpp"
#include "utils/utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestDPP{
public:
  TestDPP(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  int num_frames;
  imageGenerator generator;
  double reinit_rate;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestDPP::TestDPP(string _firstFrameFilename, string _gtFilename){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  num_particles = _num_particles;
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

void TestDPP::run(){
  dpp_tracker filter();
  Rect ground_truth;
  Mat current_frame; 
  string current_gt;
  reinit_rate = 0.0;
  time_t start, end;
  time(&start);
  Performance performance;
  namedWindow("Tracker");
  for(int k=0;k <num_frames;++k){
    current_gt=gt_vec[k];
    ground_truth=generator.stringToRect(current_gt);
    current_frame = images[k].clone();
    if(!filter.is_initialized()){
        filter.initialize(current_frame,ground_truth);
    }else{
        filter.predict();
        filter.update(current_frame,ground_truth);
    }
    imshow("Tracker",current_frame);
    waitKey(1);
  }
  time(&end);
  double sec = difftime (end, start);
};

int main(int argc, char* argv[]){
    
    
    if(argc != 7) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
        int _num_particles;
        if(strcmp(argv[1], "-img") == 0) {
            _firstFrameFilename=argv[2];
        }
        else{
            cerr <<"No images given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[3], "-gt") == 0) {
            _gtFilename=argv[4];
        }
        else{
            cerr <<"No ground truth given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        
        TestDPP tracker(_firstFrameFilename,_gtFilename);
        tracker.run();
    }
}

