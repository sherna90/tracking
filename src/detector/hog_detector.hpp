#ifndef HOG_DETECTOR_H
#define HOG_DETECTOR_H
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
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
    int padding;
} ;

class HOGDetector
{
public:
	HOGDetector();
	void generateFeatures(Mat &frame, double label);
	void generateFeature(Mat &frame, double label);
	void dataClean();
	void draw();
	MatrixXd getFeatures();
	Args args;
	VectorXd getDetectionWeights();
	void saveToCSV(string name, bool append = true);
	void loadFeatures(MatrixXd features, VectorXd labels);
	virtual MatrixXd getFeatureValues(Mat &current_frame){cout << "Error, 'getFeatureValues' function, not established" << endl;};
protected:
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	int n_descriptors, n_data;
	vector<Rect> detections;
	VectorXd labels;
	VectorXd weights, penalty_weights;
	Mat frame;
	C_utils tools;
	mt19937 generator;
};

#endif
