#ifndef BERNOULLI_PARTICLE_FILTER_H
#define BERNOULLI_PARTICLE_FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <time.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>


#include "../likelihood/gaussian.hpp"
#include "../likelihood/multivariate_gaussian.hpp"
#include "../likelihood/hmc.hpp"
#include "../likelihood/logistic_regression.hpp"
#include "../features/local_binary_pattern.hpp"
#include "../DPP/dpp.hpp"
#include "../utils/utils.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

typedef struct particle {
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
} particle;

class BernoulliParticleFilter{
public:
	int n_particles;
	~BernoulliParticleFilter();
	BernoulliParticleFilter();
	BernoulliParticleFilter(int n_particles, double lambda, double mu, double epsilon);
	bool is_initialized();
	void initialize(const Mat& current_frame, const Rect ground_truth);
	void reinitialize();
	void predict();
	void update(const Mat& image);
	void draw_particles(Mat& image, Scalar color = Scalar(0, 255, 255));
	void draw_dpp(Mat& image, Scalar color = Scalar(255,0,0));
	void resample();
	Rect estimate(const Mat& image, bool draw = false);

private:
	bool initialized;
	Rect reference_roi;
	vector<particle> states;
	vector<double> weights;
	vector<Rect> estimates;
	//VectorXd weights;
	vector<VectorXd> theta_x;
	Size img_size;
	mt19937 generator;
	float ESS;
	//vector<Rect> sampleBox;
	double existence_prob, new_existence_prob;
	double lambda, mu, epsilon;
	int step_slide;
	DPP dpp;
	LocalBinaryPattern local_binary_pattern;
	PCA pca;
	Hamiltonian_MC hamiltonian_monte_carlo;
	LogisticRegression logistic_regression;
	MatrixXd featureValues;
	VectorXd reference_hist;
	VectorXd intersectionArea;
	vector<Rect> dppResults, preDetections;
};

#endif