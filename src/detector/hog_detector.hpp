#ifndef HOG_DETECTOR_H
#define HOG_DETECTOR_H
#include <Eigen/Dense>
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
#include "../features/hist.hpp"
#include "../DPP/dpp.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

struct Args {
	bool make_gray = true;
    bool resize_src = true;
    int width, height;
    int hog_width, hog_height;
    double scale;
    int nlevels;
    double gr_threshold;
    double hit_threshold;
    bool hit_threshold_auto;
    int win_width;
    int test_stride_width, test_stride_height;
    int train_stride_width, train_stride_height;
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
	void saveToCSV(string name, bool append = true);
	void loadFeatures(MatrixXd features, VectorXd labels);
	virtual MatrixXd getFeatureValues(Mat &current_frame){
	MatrixXd void_features;
	return void_features;
	};
protected:
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	int n_descriptors, n_data;
	vector<Rect> detections;
	VectorXd labels;
	vector<double> weights;
	Mat frame;
	C_utils tools;
	mt19937 generator;
};

#endif
