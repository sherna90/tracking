#ifndef CPU_HMC_HOG_DETECTOR_H
#define CPU_HMC_HOG_DETECTOR_H
#include "hog_detector.hpp"
#include "../likelihood/CPU_hmc.hpp"
#include "../libs/piotr/gradientMex.hpp"

class CPU_HMC_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold,Rect reference_roi);
	vector<Rect> detect(Mat &frame,Rect reference_roi);
	vector<double> detect(Mat &frame,vector<Rect> samples);
	void train(Mat &frame,Rect reference_roi);
	//void train();
	//VectorXd predict(MatrixXd data);
	MatrixXd getFeatureValues(Mat &current_frame);
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
protected:
	HOGDescriptor hog;
	CPU_Hamiltonian_MC hmc;
	int num_frame;
};

#endif
