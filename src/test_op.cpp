#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "utils/c_utils.hpp"
#include "libs/piotr/gradientMex.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

VectorXd genHog(Mat &frame)
{ 
  /*int interpolation;
  if(args.hog_width > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }*/
  Mat current_frame,mat_hog_features;
  frame.copyTo(current_frame);//
  //resize(frame,current_frame,Size(args.hog_width, args.hog_height),0,0,interpolation);
  
  current_frame.convertTo( current_frame, CV_32FC(3), 1.0/255.0); //to double
  //current_frame.convertTo( current_frame, CV_32FC(3));
  current_frame *= 255.;
  piotr::fhogToCol(current_frame,mat_hog_features,8,0,0);
  VectorXd hog_features=VectorXd::Zero(mat_hog_features.cols);
  for (int j = 0; j < mat_hog_features.cols; j++){
    hog_features(j) =mat_hog_features.at<float>(0,j);
  }
  hog_features.normalize();
  return hog_features;
}

VectorXd genRawPixels(Mat &frame)
{
  /*int interpolation;
  if(args.hog_width/2 > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }*/
  Mat current_frame;
  frame.copyTo(current_frame);
  current_frame.convertTo( current_frame, CV_32FC(3), 1.0/255.0); //to double
  C_utils tools;
  Mat cieLabFrame = tools.RGBtoLAB(current_frame);

  //resize(cieLabFrame,cieLabFrame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);

  int channels = cieLabFrame.channels();
  vector<Mat> frame_channels(channels);
  split(cieLabFrame, frame_channels);
  VectorXd rawPixelsFeatures(cieLabFrame.cols*cieLabFrame.rows*channels);
  int cidx=0;
  for (int ch = 0; ch < channels; ++ch){   
      for(int c = 0; c < cieLabFrame.cols ; c++){
        for(int r = 0; r < cieLabFrame.rows ; r++){
            rawPixelsFeatures(cidx) = (double)frame_channels[ch].at<double>(r,c);
            cidx++;
        }
    }
  }
  double normTerm = rawPixelsFeatures.norm();
  if (normTerm > 1e-6){
    rawPixelsFeatures.normalize();
    //rawPixelsFeatures =  rawPixelsFeatures.array()/normTerm;
  }
  return rawPixelsFeatures;
}

int main()
{
  C_utils utils;
  Mat current_frame = imread("mario_resized.jpg");
  cvtColor(current_frame, current_frame, COLOR_RGB2BGR);
  
  /*int channels = current_frame.channels();
  vector<Mat> frame_channels(channels);
  split(current_frame, frame_channels);
  for (int ch = 0; ch < channels; ++ch) cout << frame_channels[ch] << endl;*/

  VectorXd b = genHog(current_frame);

  VectorXd a = genRawPixels(current_frame);

  MatrixXd ba(2,b.rows()+a.rows());
  VectorXd c(b.rows()+a.rows());
  c <<b,a;
  ba.row(0) = c;
  cout<< "here"<< endl;
  ba.row(1) = VectorXd::Ones(b.rows()+a.rows());

  VectorXd pdataNorm = ba.rowwise().squaredNorm().array().sqrt();
  ba = ba.array().colwise() / pdataNorm.array();
  
  cout << ba.transpose() << endl;

  return 0;
}

