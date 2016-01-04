#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../include/pmmh.hpp"
#include "../include/utils.hpp"
#include "../include/image_generator.hpp"
#include "ct/CompressiveTracker.h"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestCT{
public:
  TestCT(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  int num_frames;
  imageGenerator generator;
  double reinit_rate;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestCT::TestCT(string _firstFrameFilename, string _gtFilename){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

void TestCT::run(){
  CompressiveTracker ct=CompressiveTracker();
  Rect ground_truth,estimate;
  Mat current_frame,grayImg; 
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
    cvtColor(current_frame, grayImg, CV_RGB2GRAY);
    if(k==0){
      ct.init(grayImg, ground_truth);
      estimate=ground_truth; 
    }
    else{
      ct.processFrame(grayImg, estimate);
      rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
      rectangle( current_frame, estimate, Scalar(0,0,255), 1, LINE_AA );
      double r1 = performance.calc(ground_truth, estimate);
      if(r1<0.1) {
        ct.init(grayImg, ground_truth); 
        reinit_rate+=1.0;
      }
    }
    imshow("Tracker",current_frame);
    waitKey(25);
  }
  time(&end);
  double sec = difftime (end, start);
  // print precision,recall,fps,rate,num_frames
  cout  << performance.get_avg_precision()/(num_frames-reinit_rate);
  cout << "," << performance.get_avg_recall()/(num_frames-reinit_rate);
  cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
    if(argc != 5) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
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
        
        TestCT tracker(_firstFrameFilename,_gtFilename);
        tracker.run();
    }
}

