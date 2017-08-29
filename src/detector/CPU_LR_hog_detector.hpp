#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include "hog_detector.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"

class CPU_LR_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold,Rect reference_roi);
	vector<Rect> detect(Mat &frame,Rect reference_roi);
	vector<Rect> detect(Mat &frame,vector<Rect> samples);
	void train(Mat &frame,Rect reference_roi);
	void train();
	//VectorXd predict(MatrixXd data);
	MatrixXd getFeatureValues(Mat &current_frame);
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	void samplerBox(Mat &current_frame, Rect ground_truth, int n_particles, vector<Rect>& sampleBox, vector<Rect>& negativeBox);
protected:
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
	int num_frame;
};

#endif
