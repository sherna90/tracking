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
    bool make_gray;
    bool resize_src;
    int hog_width;
    int hog_height;
    double gr_threshold;
    double hit_threshold;
    int n_orients;
    int bin_size;
    double overlap_threshold;
    double p_accept;
    double lambda;
    double alpha;
    double step_size;
    int n_iterations;
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
    vector<double> getWeights();
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
