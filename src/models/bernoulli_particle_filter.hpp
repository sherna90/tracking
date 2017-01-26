#ifndef BERNOULLI_PARTICLE_FILTER_H
#define BERNOULLI_PARTICLE_FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fftw3.h>

#include "../features/haar.hpp"

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
	BernoulliParticleFilter(int n_particles);
	bool is_initialized();
	void initialize(Mat& current_frame, Rect ground_truth);
	void reinitialize();
	void predict();
	void update(Mat& image);
	void draw_particles(Mat& image, Scalar color = Scalar(0, 255, 255));
	Rect estimate(Mat& image, bool draw = false);

private:
	bool initialized;
	vector<particle> states;
	//vector<float> weights;
	VectorXd weights;
	vector<VectorXd> theta_x;
	Rect reference_roi;
	Size img_size;
	mt19937 generator;
	float ESS;
	vector<Rect> sampleBox;
	Haar haar;
	double existence_prob;
};

#endif