#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include "hog_detector.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"
#include "../libs/piotr_fhog/fhog.hpp"
#include "../libs/cn/cnfeat.hpp"

class CPU_LR_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold,Rect reference_roi);
	vector<Rect> detect(Mat &frame,Rect reference_roi);
	vector<double> detect(Mat &frame,vector<Rect> samples);
	void train(Mat &frame,Rect reference_roi);
	void train();
	MatrixXd getFeatureValues(Mat &current_frame);
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
protected:
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
	int num_frame=0;
	double max_value=1.0;
};

#endif
