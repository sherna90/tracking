#include "dpp_tracker.hpp"

#ifndef PARAMS
const double RATIO_TRAINTEST = 0.8;
const float OVERLAP_THRESHOLD = 0.8;
const int STEPSLIDE = 5;

//DPP's parameters
const double ALPHA = 0.9;
const double LAMBDA = -0.1;
const double BETA = 1.1;
const double MU = 0;
const double EPSILON = 0.4;

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
	this->dpp = DPP();
	cvtColor(current_frame, grayImg, CV_RGB2GRAY);
	int left = MAX(ground_truth.x, 1);
	int top = MAX(ground_truth.y, 1);
	int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
	int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
	Rect reference_roi = Rect(left, top, right - left, bottom - top);
	this->weights.resize(0);
	for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
		for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
			Rect current_window(col, row, reference_roi.width, reference_roi.height);
			Rect intersection = reference_roi & current_window;
			this->detections.push_back(current_window);
			this->weights.conservativeResize( this->weights.size() + 1 );
			this->weights(this->weights.size() - 1) = intersection.area();
		}
	}
	
   	this->featureValues = MatrixXd(this->haar.featureNum, this->detections.size());
   	this->haar.init(grayImg, reference_roi, this->detections);
   	cv2eigen(this->haar.sampleFeatureValue, this->featureValues);
   	this->featureValues.transposeInPlace();
	this->initialized = true;
	cout << "initialized!!!" << endl;
	cout << this->detections.size() << endl;
}

void DPPTracker::predict(){
  	this->dppResults = this->dpp.run(this->detections, this->weights, this->featureValues, ALPHA, LAMBDA, BETA, MU, EPSILON);
}

void DPPTracker::update(Mat& image){
	Mat grayImg;
	this->weights.resize(0);
	this->detections.clear();
	cvtColor(image, grayImg, CV_RGB2GRAY);
	/*int left = MAX(ground_truth.x, 1);
	int top = MAX(ground_truth.y, 1);
	int right = MIN(ground_truth.x + ground_truth.width, image.cols - 1);
	int bottom = MIN(ground_truth.y + ground_truth.height, image.rows - 1);
	reference_roi = Rect(left, top, right - left, bottom - top);*/
	Rect reference_roi;
	for (size_t i = 0; i < this->dppResults.size(); ++i)
	{
		Rect box = this->dppResults.at(i);
		reference_roi.x += box.x; reference_roi.y += box.y;
  		reference_roi.height += box.height; reference_roi.width += box.width;
	}
	reference_roi.x /= this->dppResults.size(); reference_roi.y /= this->dppResults.size();
  	reference_roi.height /= this->dppResults.size(); reference_roi.width /= this->dppResults.size();

	for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
    	for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
      		Rect current_window(col, row,  reference_roi.width, reference_roi.height);
			Rect intersection = reference_roi & current_window;
			this->detections.push_back(current_window);
			this->weights.conservativeResize( this->weights.size() + 1 );
			this->weights(this->weights.size() - 1) = intersection.area();
    	}
  	}

	this->featureValues = MatrixXd(this->haar.featureNum, this->detections.size());
	this->haar.getFeatureValue(grayImg, this->detections);
	cv2eigen(this->haar.sampleFeatureValue,this->featureValues);
	this->featureValues.transposeInPlace();

}

Rect DPPTracker::estimate(Mat& image, bool draw){
  	if(draw){
    	for (size_t i = 0; i < this->dppResults.size(); ++i)
    	{
			Rect box = this->dppResults.at(i);
			rectangle( image, box, Scalar(0,0,255), 2, LINE_AA );
    	}
  	}

  	Rect estimate;
  	for (size_t i = 0; i < this->dppResults.size(); ++i)
  	{
  		Rect box = this->dppResults.at(i);
  		estimate.x += box.x; estimate.y += box.y;
  		estimate.height += box.height; estimate.width += box.width;
  	}
  	estimate.x /= this->dppResults.size(); estimate.y /= this->dppResults.size();
  	estimate.height /= this->dppResults.size(); estimate.width /= this->dppResults.size();

  	rectangle( image, estimate, Scalar(255,0,0), 2, LINE_AA);

  	return estimate;
}
