#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "hog_detector.hpp"
#include "../features/hist.hpp"
#include "../DPP/dpp.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"
typedef struct Roi {
    float x; /** current x coordinate */
    float y; /** current y coordinate */
    float width; /** current width coordinate */
    float height; /** current height coordinate */
    float scale; /** current velocity bounding box scale */
    float x_p; /** current x coordinate */
    float y_p; /** current y coordinate */
    float width_p; /** current width coordinate */
    float height_p; /** current height coordinate */
    float scale_p; /** current velocity bounding box scale */
} Roi;
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
protected:
	HOGDescriptor hog;
	mt19937 generator;
	CPU_LogisticRegression logistic_regression;
	int num_frame;
};

#endif
