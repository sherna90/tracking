//#include "../include/particle_filter.hpp"
#include "../include/utils.hpp"
#include "../include/image_generator.hpp"
#include "../include/mb_lbp.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
//#include "haar.hpp"
//#include "gaussian.hpp"
#include <time.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;

class TestMethods{
public:
  TestMethods(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  //int num_particles,num_frames;
  int num_frames;
  imageGenerator generator;
  //double reinit_rate;
  //discrete_particle_filter filter;
  //particle_filter filter;
  vector<Mat> images;
  vector<string> gt_vec;
};

TestMethods::TestMethods(string _firstFrameFilename, string _gtFilename){
  imageGenerator generator(_firstFrameFilename,_gtFilename);
  //num_particles = _num_particles;
  num_frames = generator.getDatasetSize();
  gt_vec = generator.ground_truth;
  images = generator.images;
}

double KullbackLeibler_divergence( VectorXd p, VectorXd q){
	double divergence = 0.0;
	for (int i = 0; i < p.rows(); ++i){
		if ((p(i) != 0.0) && (q(i) != 0.0)){
			divergence += (p(i) * log2(p(i) / q(i))); 
		}
	}
	return divergence;
}	

void TestMethods::run(){
  /*particle_filter filter(num_particles);*/
  Rect ground_truth;
  Mat current_frame; 
  string current_gt;
  /*reinit_rate = 0.0;
  time_t start, end;
  time(&start);
  Performance performance;*/

  int p_blocks = 3, n_features = 59, slider = 3;
  bool copy_border = false;
  bool multiscale = true;
  int multiscale_slider = 3;
  int n_scales = 4;
  int region = 60;
  double threshold = 0.09;
  //vector<int> region_scales = { 0, -40, -20, -15, -10, -5, 5, 10, 15, 20, 40};
  vector<int> region_scales = {0};

  namedWindow("Test");
  for(int k=0;k <num_frames;++k){

    current_gt=gt_vec[k];
    ground_truth=generator.stringToRect(current_gt);
    current_frame = images[k].clone();
    vector<Rect> targetBox;
    targetBox.push_back(ground_truth);

    /*if(!filter.is_initialized()){
        filter.initialize(current_frame,ground_truth);
    }else{
        filter.predict();
        filter.update(current_frame);
        filter.draw_particles(current_frame,Scalar(0,255,255));
        rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
        Rect estimate = filter.estimate(current_frame,true);
        double r1 = performance.calc(ground_truth, estimate);
        //cout  << "ESS : " << filter.getESS() << "ratio : " << r1 << endl;
        if(r1<0.1) {
          filter.reinitialize();
          reinit_rate+=1.0;
      }
    }*/

    Mat grayImg;
    cvtColor(current_frame, grayImg, CV_RGB2GRAY);
    equalizeHist( grayImg, grayImg );

    MultiScaleBlockLBP MBLBP_train(p_blocks,n_features,slider,copy_border, multiscale, multiscale_slider, n_scales);
    MBLBP_train.init(grayImg, targetBox);

    //exit(0);
    rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );

    int xsize = current_frame.cols;
    int ysize = current_frame.rows;

    vector<Rect> slidingWindows;
    for (unsigned int s = 0; s < region_scales.size(); ++s){
    	int x = 1;
    	int y = 1;
    	while((y+(region+region_scales.at(s))+slider)<= ysize+1){
	    	while ((x+(region+region_scales.at(s))+slider) <= xsize+1){
	    		Rect window(x,y,region+ region_scales.at(s),region+ region_scales.at(s));
	    		slidingWindows.push_back(window);
	    		x+=slider;
	    	}
	    	y+=slider;
	    	x=1;
    	}
    }



    MultiScaleBlockLBP MBLBP_test(p_blocks,n_features,slider,copy_border, multiscale, multiscale_slider, n_scales);
    MBLBP_test.init(grayImg, slidingWindows);

    for (unsigned int i = 0; i < slidingWindows.size(); ++i){
    	//double metric = KullbackLeibler_divergence(MBLBP_test.sampleFeatureValue.row(i), MBLBP_train.sampleFeatureValue.row(k));
    	double metric = bhattarchaya(MBLBP_test.sampleFeatureValue.row(i), MBLBP_train.sampleFeatureValue.row(k));
    	//cout << metric << endl;
    	if (metric <= threshold){
    		rectangle( current_frame, slidingWindows.at(i), Scalar(0,255,255), 1, LINE_AA );
    	}
    }

    imshow("Test",current_frame);
  	waitKey(100000);
  }


  /*time(&end);
  double sec = difftime (end, start);*/
  // print precision,recall,fps,rate,num_frames
  //cout << "ML:" << filter.getMarginalLikelihood() << endl;
  /*cout  << performance.get_avg_precision()/(num_frames-reinit_rate);
  cout << "," << performance.get_avg_recall()/(num_frames-reinit_rate);
  cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;*/
};

int main(int argc, char* argv[]){
    
    
    //if(argc != 7) {
    if(argc != 5) {	
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
        //int _num_particles;
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
        /*if(strcmp(argv[5], "-npart") == 0) {
            _num_particles=atoi(argv[6]);
        }
        else{
             _num_particles=300;
         }*/
        TestMethods MBLBP(_firstFrameFilename,_gtFilename);
        MBLBP.run();
    }
}

