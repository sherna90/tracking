#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include <opencv2/imgproc.hpp>
#include "hog_detector.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"


class CPU_LR_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold);
	void init(double group_threshold, double hit_threshold, Rect reference_roi);
	vector<Rect> detect(Mat &frame);
	void train(Mat &frame,Rect reference_roi);
	void train();
	VectorXd predict(MatrixXd data);
	MatrixXd getFeatureValues(Mat &current_frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
protected:
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
};

#endif
