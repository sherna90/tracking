#ifndef CPU_HOG_DETECTOR_H
#define CPU_HOG_DETECTOR_H
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include "../likelihood/logistic_regression.hpp"
#include "../utils/c_utils.hpp"
#include "../DPP/nms.hpp"

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
    int padding;
} ;

class CPU_HOGDetector
{
public:
	CPU_HOGDetector();
	CPU_HOGDetector(double group_threshold, double hit_threshold);
	CPU_HOGDetector(double group_threshold, double hit_threshold, Rect reference_roi);
	vector<Rect> detect(Mat &frame);
	void train(Mat &frame,Rect reference_roi);
	void train();
	VectorXd predict(MatrixXd data);
	void draw();
	void generateFeatures(Mat &frame, double label);
	void generateFeature(Mat &frame, double label);
	MatrixXd getFeatureValues(Mat &current_frame);
	MatrixXd getFeatures();
	VectorXd getDetectionWeights();
	void dataClean();
	Args args;
	MatrixXd compute(Mat &frame);
	void saveToCSV(string name, bool append = true);
	void loadFeatures(MatrixXd features, VectorXd labels);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
private:
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	HOGDescriptor hog;
	int n_descriptors, n_data;
	vector<Rect> detections;
	VectorXd labels;
	VectorXd weights;
	Mat frame;
	C_utils tools;
	LogisticRegression logistic_regression;
	//Hamiltonian_MC hmc;
	mt19937 generator;
};

#endif
