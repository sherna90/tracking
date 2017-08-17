#ifndef Mask_CPU_HMC_HOG_DETECTOR_H
#define Mask_CPU_HMC_HOG_DETECTOR_H
#include <opencv2/imgproc.hpp>
#include "hog_detector.hpp"
#include "../likelihood/Mask_CPU_hmc.hpp"


class Mask_CPU_HMC_HOGDetector : public HOGDetector
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
	Mask_CPU_Hamiltonian_MC hmc;
};

#endif
