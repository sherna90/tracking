#include "dpp_tracker.hpp"

#ifndef PARAMS
const double RATIO_TRAINTEST = 0.8;
const float OVERLAP_THRESHOLD = 0.5;
const int STEPSLIDE = 5;

//DPP's parameters
const double LAMBDA = 0.9;
const double MU = 0.5;
const double EPSILON = 0.5;

#endif

DPPTracker::DPPTracker(){
	this->initialized = false;
}

DPPTracker::~DPPTracker(){
	this->detections.clear();
	this->weights.resize(0);
	this->labels.resize(0);
}

bool DPPTracker::is_initialized() {
    return this->initialized;
}

void DPPTracker::initialize(Mat& current_frame, Rect ground_truth){
	Mat grayImg;
	image_size = current_frame.size();
	this->detections.clear();
	this->weights.resize(0);
	this->labels.resize(0);
	this->dpp = DPP();
	cvtColor(current_frame, grayImg, CV_RGB2GRAY);
	int left = MAX(ground_truth.x, 1);
	int top = MAX(ground_truth.y, 1);
	int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
	int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
	reference_roi = Rect(left, top, right - left, bottom - top);
	for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
		for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
			Rect current_window(col, row, reference_roi.width, reference_roi.height);
			Rect intersection = reference_roi & current_window;
			this->detections.push_back(current_window);
			this->weights.conservativeResize( this->weights.size() + 1 );
			this->labels.conservativeResize( this->labels.size() + 1 );
			this->weights(this->weights.size() - 1) = (float)intersection.area()/reference_roi.area();
			this->labels(this->labels.size() - 1) = ((float)intersection.area()/reference_roi.area()>OVERLAP_THRESHOLD) ? 1.0 : -1.0;
		}
	}
    MatrixXd sample_feature_values;
    this->local_binary_pattern.init(grayImg, this->detections);
    this->featureValues = MatrixXd(this->detections.size(), this->local_binary_pattern.sampleFeatureValue.cols());
   	this->featureValues << this->local_binary_pattern.sampleFeatureValue;
    this->initialized = true;
	cout << "initialized!!!" << endl;
	cout << "Positive examples = " << (this->labels.array() > 0).count() << endl;
	cout << "Negative examples = " << (this->labels.array() < 0).count() << endl;
	this->logistic_regression = LogisticRegression(this->featureValues, labels);
    this->logistic_regression.train(1e4,1e-3,1e-2);
    phi = this->weights;
}

void DPPTracker::predict(){
   	VectorXd qualityTerm;
   	this->dppResults = this->dpp.run(this->detections,this->phi, this->weights, this->featureValues, qualityTerm, LAMBDA, MU, EPSILON);
}

void DPPTracker::update(Mat& image){
	Mat grayImg;
	cvtColor(image, grayImg, CV_RGB2GRAY);
	this->labels.resize(0);
	for(unsigned int i=0;i<this->dppResults.size();i++){
		Rect current_window=this->dppResults.at(i);
		Rect intersection = reference_roi & current_window;
		this->labels.conservativeResize( this->labels.size() + 1 );
		this->labels(this->labels.size() - 1) = ((float)intersection.area()>OVERLAP_THRESHOLD) ? 1.0 : -1.0;
	}
	this->local_binary_pattern.init(grayImg, this->dppResults);
    if(this->featureValues.rows() > 1){
		this->logistic_regression.setData(this->local_binary_pattern.sampleFeatureValue, this->labels);
    	this->logistic_regression.train(1e2,1e-6,1e-2);
	}
	this->weights.resize(0);
	this->detections.clear();
	this->labels.resize(0);
	for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
    	for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
      		Rect current_window(col, row,  reference_roi.width, reference_roi.height);
			Rect intersection = reference_roi & current_window;
			this->detections.push_back(current_window);
			this->weights.conservativeResize( this->weights.size() + 1 );
			this->labels.conservativeResize( this->labels.size() + 1 );
			this->weights(this->weights.size() - 1) = (float)intersection.area()/reference_roi.area();
			this->labels(this->labels.size() - 1) = ((float)intersection.area()>OVERLAP_THRESHOLD) ? 1.0 : -1.0;
    	}
  	}
	this->local_binary_pattern.init(grayImg, this->detections);
	this->featureValues = MatrixXd(this->detections.size(), this->local_binary_pattern.sampleFeatureValue.cols());
   	this->featureValues << this->local_binary_pattern.sampleFeatureValue;
	phi = this->logistic_regression.predict(this->featureValues, false);
}

void DPPTracker::draw_results(Mat& image){
	for (size_t i = 0; i < this->dppResults.size(); ++i)
	{
		Rect box = this->dppResults.at(i);
		rectangle( image, box, Scalar(0,0,255), 2, LINE_AA );
	}
}

Rect DPPTracker::estimate(Mat& image, bool draw){
  	Rect estimate;
  	float _x = 0.0, _y = 0.0, _width = 0.0, _height = 0.0, norm = 0.0;
  	
  	for (size_t i = 0; i < this->dppResults.size();i++){
        Rect state = this->dppResults.at(i);
        if(state.x > 0 && state.x < image_size.width
            && state.y > 0  && state.y < image_size.height 
            && state.width > 0 && state.width < image_size.height 
            && state.height > 0 && state.height < image_size.height){
            _x += state.x;
            _y += state.y;
            _width += state.width; 
            _height += state.height;
            norm++;
        }
    }

    Point pt1,pt2;
    pt1.x = cvRound(_x / norm);
    pt1.y = cvRound(_y / norm);
    _width = cvRound(_width / norm);
    _height = cvRound(_height / norm);
    pt2.x = cvRound(pt1.x + _width);
    pt2.y = cvRound(pt1.y + _height); 
    if(pt2.x < image_size.width && pt1.x >= 0 && pt2.y < image_size.height && pt1.y >= 0){
    	estimate = Rect( pt1.x, pt1.y, _width, _height );
    	reference_roi=estimate;
        if(draw) rectangle( image, estimate, Scalar(255,0,0), 2, LINE_AA );
    }
  	return estimate;
}
