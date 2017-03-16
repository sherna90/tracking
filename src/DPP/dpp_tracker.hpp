#ifndef DPPTRACKER_H
#define DPPTRACKER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <random>

#include "../utils/image_generator.hpp"
#include "../utils/c_utils.hpp"

#include "../features/haar.hpp"
#include "../features/hog.hpp"
#include "../features/local_binary_pattern.hpp"
#include "../features/mb_lbp.hpp"

#include "../likelihood/gaussian.hpp"
#include "../likelihood/logistic_regression.hpp"
#include "../likelihood/hamiltonian_monte_carlo.hpp"
#include "../likelihood/multinomialnaivebayes.hpp"
#include "../likelihood/incremental_gaussiannaivebayes.hpp"

#include "dpp.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

class DPPTracker{
public:
	DPPTracker();
	~DPPTracker();
	void initialize(Mat& current_frame, Rect ground_truth);
	bool is_initialized();
	void predict();
	void update(Mat& image);
	void draw_results(Mat& image);
	Rect estimate(Mat& image, bool draw = false);
private:
	bool initialized;
	mt19937 generator;
	vector<Rect> detections;
	VectorXd weights,labels,phi;
	LocalBinaryPattern local_binary_pattern;
	MatrixXd featureValues;
	Haar haar;
	DPP dpp;
	vector<Rect> dppResults;
	Size image_size;
	Rect reference_roi;
	LogisticRegression logistic_regression;
};

#endif
