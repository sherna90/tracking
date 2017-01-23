#include "dpp_tracker.hpp"

#ifndef PARAMS
const double RATIO_TRAINTEST = 0.8;
const float OVERLAP_THRESHOLD = 0.8;
const int STEPSLIDE = 1;
#endif

DPPTracker::DPPTracker(){
  this->initialized = false;
}

DPPTracker::~DPPTracker(){
  this->detections.clear();
  this->weights.resize(0);
}

bool DPPTracker::is_initialized() {
    return this->initialized;
}

void DPPTracker::initialize(Mat& current_frame, Rect ground_truth){
	Mat grayImg;
	this->detections.clear();
  cvtColor(current_frame, grayImg, CV_RGB2GRAY);
  //Size im_size = current_frame.size();
  int left = MAX(ground_truth.x, 1);
  int top = MAX(ground_truth.y, 1);
  int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
  int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
  Rect reference_roi = Rect(left, top, right - left, bottom - top);
  this->weights.resize(0); 
	
  for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
		for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
			Rect current_window(col, row, reference_roi.height, reference_roi.width);
			Rect intersection = reference_roi & current_window;
			this->detections.push_back(current_window);
			this->weights.conservativeResize( this->weights.size() + 1 );
			this->weights(this->weights.size() - 1) = intersection.area();
		}
	}
	
  this->featureValues = MatrixXd(haar.featureNum, this->detections.size());
	haar.init(grayImg, reference_roi, this->detections);
	cv2eigen(haar.sampleFeatureValue,this->featureValues);
	initialized = true;
	cout << "initialized!!!" << endl;
}

void DPPTracker::predict(){
}

void DPPTracker::update(Mat& image, Rect ground_truth){
}

Rect DPPTracker::estimate(Mat& image, bool draw){

}
