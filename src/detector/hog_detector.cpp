#include "hog_detector.hpp"
HOGDetector::HOGDetector(){
	
}

void HOGDetector::generateFeatures(Mat &frame, double label)
{	
	MatrixXd hogFeatures = this->getFeatureValues(frame);
	MatrixXd oldhogFeatures = this->feature_values;
	this->feature_values.conservativeResize(this->feature_values.rows() + hogFeatures.rows(), NoChange);
	this->feature_values << oldhogFeatures, hogFeatures;
	VectorXd hogLabels = VectorXd::Ones(feature_values.rows())*label;
	VectorXd oldhogLabels = this->labels;
	this->labels.conservativeResize(this->labels.rows() + hogLabels.size());
	this->labels << oldhogLabels, hogLabels;
}

void HOGDetector::generateFeature(Mat &frame, double label)
{
	this->feature_values= this->getFeatureValues(frame);
	this->labels = VectorXd::Ones(feature_values.rows())*label;
}

void HOGDetector::dataClean(){
	this->feature_values.resize(0,0);
	this->labels.resize(0);
}

void HOGDetector::draw()
{
	for (size_t i = 0; i < this->detections.size(); i++)
    {
        Rect r = this->detections[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(this->frame, r.tl(), r.br(), cv::Scalar(255,0,0), 3);
    }
}

MatrixXd HOGDetector::getFeatures()
{
	return this->feature_values;
}

VectorXd HOGDetector::getDetectionWeights(){
	return this->weights;
}

void HOGDetector::saveToCSV(string name, bool append){
	tools.writeToCSVfile(name+"_values.csv", this->feature_values, append);
	tools.writeToCSVfile(name+"_labels.csv", this->labels, append);
}

void HOGDetector::loadFeatures(MatrixXd features, VectorXd labels){
	this->dataClean();
	this->feature_values = features;
	this->labels = labels;
}