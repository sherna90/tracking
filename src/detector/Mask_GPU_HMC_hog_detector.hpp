#ifndef Mask_GPU_HMC_HOG_DETECTOR_H
#define Mask_GPU_HMC_HOG_DETECTOR_H
#include <opencv2/cudaobjdetect.hpp>
#include "hog_detector.hpp"
#include "../likelihood/Mask_GPU_hmc.hpp"

class Mask_GPU_HMC_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold);
	void init(double group_threshold, double hit_threshold, Rect reference_roi);
	vector<Rect> detect(Mat &frame);
	void train(Mat &frame,Rect reference_roi);
	void train();
	VectorXd predict(MatrixXd data);
	MatrixXd getFeatureValues(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
protected:
	Ptr<cuda::HOG> gpu_hog;
	Mask_GPU_Hamiltonian_MC hmc;
};

#endif
