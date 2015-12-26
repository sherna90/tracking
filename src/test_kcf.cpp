#include "../../KCFcpp/src/kcftracker.hpp"
#include "../include/utils.hpp"
#include "../include/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestKCF{
public:
  TestKCF(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  int num_frames;
  imageGenerator generator;
  double reinit_rate;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestKCF::TestKCF(string _firstFrameFilename, string _gtFilename){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

void TestKCF::run(){
  KCFTracker tracker(true, true, true, false);
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
    if(k==0){
        tracker.init(ground_truth,current_frame);
    }else{
        Rect estimate = tracker.update(current_frame);
        rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
        rectangle( current_frame, estimate, Scalar(0,0,255), 1, LINE_AA );
        double r1 = performance.calc(ground_truth, estimate);
        if(r1<0.1) {
          tracker.init(ground_truth,current_frame);
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
        else{
            cerr <<"No ground truth given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        TestKCF tracker(_firstFrameFilename,_gtFilename);
        tracker.run();
    }
}

