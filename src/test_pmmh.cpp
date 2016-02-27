#include "../include/pmmh.hpp"
#include "../include/utils.hpp"
#include "../include/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestPMMH{
public:
  TestPMMH(string _firstFrameFilename, string _gtFilename, int _num_particles,int _lag, int _mcmc);
  void run();
private:
  int num_particles,num_frames;
  int lag,mcmc;
  imageGenerator generator;
  double reinit_rate;
  particle_filter filter;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestPMMH::TestPMMH(string _firstFrameFilename, string _gtFilename, int _num_particles,int _lag, int _mcmc){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  num_particles = _num_particles;
  mcmc=_mcmc;
  lag=_lag;
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

void TestPMMH::run(){
  pmmh filter(num_particles,lag,mcmc);
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
    }else if(filter.is_initialized()){
        filter.predict();
        filter.update(current_frame);
        //filter.draw_particles(current_frame);
        rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
        Rect estimate = filter.estimate(current_frame,true);
        double r1 = performance.calc(ground_truth, estimate);
        if(r1<0.1) {
          filter.reinitialize();
          reinit_rate+=1.0;
      }
    }
    imshow("Tracker",current_frame);
    waitKey(1);
  }
  time(&end);
  double sec = difftime (end, start);
  // print precision,recall,fps,rate,num_frames
  cout  << performance.get_avg_precision()/(num_frames-reinit_rate);
  cout << "," << performance.get_avg_recall()/(num_frames-reinit_rate);
  cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
    if(argc != 11) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
        int _num_particles,_lag,_mcmc;
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
        if(strcmp(argv[5], "-npart") == 0) {
            _num_particles=atoi(argv[6]);
        }
        else{
            _num_particles=300;
        }
        if(strcmp(argv[7], "-lag") == 0) {
            _lag=atoi(argv[8]);
        }
        else{
            _lag=3;
        }
        if(strcmp(argv[9], "-mcmc") == 0) {
            _mcmc=atoi(argv[10]);
        }
        else{
            _mcmc=3;
        }
        TestPMMH tracker(_firstFrameFilename,_gtFilename,_num_particles,_lag,_mcmc);
        tracker.run();
    }
}

