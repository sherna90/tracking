#ifndef CUDA_HOG_DETECTOR_H
#define CUDA_HOG_DETECTOR_H
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <chrono>
//#include "../likelihood/GPU_hmc.hpp"
#include "../likelihood/GPU_logistic_regression.hpp"
//#include "../likelihood/logistic_regression.hpp"
#include "../utils/c_utils.hpp"
#include "../DPP/nms.hpp"
#include "../DPP/dpp.hpp"
using namespace cv;
using namespace std;
using namespace Eigen;

struct Args {
	bool make_gray = true;
    bool resize_src = true;
    int width, height;
    double scale;
    int nlevels;
    double gr_threshold;
    double hit_threshold;
    bool hit_threshold_auto;
    int win_width;
    int win_stride_width, win_stride_height;
    int block_width;
    int block_stride_width, block_stride_height;
    int cell_width;
    int nbins;
    bool gamma_corr;
    double overlap_threshold;
    double p_accept;
    double lambda, epsilon, tolerance;
    int n_iterations;
    double padding;
} ;

class CUDA_HOGDetector
{
public:
	CUDA_HOGDetector();
	CUDA_HOGDetector(double group_threshold, double hit_threshold);
	CUDA_HOGDetector(double group_threshold, double hit_threshold, Rect reference_roi);
	vector<Rect> detect(Mat &frame);
	void train(Mat &frame,Rect reference_roi);
	void draw();
	MatrixXd getFeatureValues(Mat &frame);
	MatrixXd getFeatureValues();
	VectorXd getDetectionWeights();
	void generateFeatures(Mat &frame, double label);
	void generateFeature(Mat &frame, double label);
	void dataClean();
	void train();
	VectorXd predict(MatrixXd data);
	void saveToCSV(string name, bool append = true);
	void loadFeatures(MatrixXd features, VectorXd labels);
	Args args;
	MatrixXd compute(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
private:
	int n_descriptors, n_data;
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	Ptr<cuda::HOG> gpu_hog;
	vector<Rect> detections;
	VectorXd labels;
	VectorXd weights,penalty_weights;
	Mat frame;
	C_utils tools;
	LogisticRegression logistic_regression;
	//Hamiltonian_MC hmc;
	mt19937 generator;
};

#endif
