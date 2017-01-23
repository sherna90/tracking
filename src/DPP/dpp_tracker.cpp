#include "dpp_tracker.hpp"

#ifndef PARAMS
const double RATIO_TRAINTEST = 0.8;
const float OVERLAP_THRESHOLD = 0.8;
const int STEPSLIDE = 1;
#endif


DPPTracker::DPPTracker(){
}

bool DPPTracker::is_initialized() {
    return initialized;
}

void DPPTracker::initialize(Mat& current_frame, Rect ground_truth){
	Mat grayImg;
	detections.clear();
    cvtColor(current_frame, grayImg, CV_RGB2GRAY);
    Size im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    Rect reference_roi=Rect(left, top, right - left, bottom - top);
    weights.resize(0); 
  	for(int row = 0; row <= grayImg.rows - reference_roi.rows; row+=STEPSLIDE){
  		for(int col = 0; col <= grayImg.cols - reference_roi.cols; col+=STEPSLIDE){
  			Rect current_window(col, row, reference_roi.rows, reference_roi.cols);
  			Rect intersection = reference_roi & current_window;
  			detections.push_back(current_window);
  			weights.conservativeResize( weights.size()+1 );
  			weights(weights.size()-1) = intersection.area();
  		}
  	}
  	featureValues = MatrixXd(haar.featureNum, detections.size());
  	haar.init(grayImg, reference_roi, detections);
  	cv2eigen(haar.sampleFeatureValue,featureValues);
  	initialized=true;
	cout << "initialized!!!" << endl;
}

void DPPTracker::predict(){
}


void DPPTracker::update(Mat& image){
}
