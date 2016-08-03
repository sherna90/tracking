#include "../include/pmmh.hpp"
#include "../include/smc_squared.hpp"
#include "../include/utils.hpp"
#include "../include/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestSMCSampler{
public:
  TestSMCSampler(string _firstFrameFilename, string _gtFilename, int _num_particles, int _num_theta,int _lag, int _mcmc);
  void run();
private:
  int num_particles,num_frames,num_theta;
  int lag,mcmc;
  imageGenerator generator;
  double reinit_rate;
  particle_filter filter;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestSMCSampler::TestSMCSampler(string _firstFrameFilename, string _gtFilename, int _num_particles,int _num_theta,int _lag, int _mcmc){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  num_particles = _num_particles;
  num_theta=_num_theta;
  mcmc=_mcmc;
  lag=_lag;
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

void TestSMCSampler::run(){
  smc_squared filter(num_particles,num_theta,lag,mcmc);
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
        filter.update(current_frame);
        filter.draw_particles(current_frame);
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
  waitKey(1);
  time(&end);
  double sec = difftime (end, start);
  cout  << performance.get_avg_precision()/(num_frames-reinit_rate);
  cout << "," << performance.get_avg_recall()/(num_frames-reinit_rate);
  cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
    if(argc != 13) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
        int _num_particles,_num_theta,_lag,_mcmc;
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
        if(strcmp(argv[7], "-ntheta") == 0) {
            _num_theta=atoi(argv[8]);
        }
        else{
            _num_theta=300;
        }
        if(strcmp(argv[9], "-lag") == 0) {
            _lag=atoi(argv[10]);
        }
        else{
            _lag=3;
        }
        if(strcmp(argv[11], "-mcmc") == 0) {
            _mcmc=atoi(argv[12]);
        }
        else{
            _mcmc=3;
        }
        TestSMCSampler tracker(_firstFrameFilename,_gtFilename,_num_particles,_num_theta,_lag,_mcmc);
        tracker.run();
    }
}

