#ifndef TEST_BERNOULLI_PARTICLE_FILTER_H
#define TEST_BERNOULLI_PARTICLE_FILTER_H

#include "models/bernoulli_particle_filter.hpp"
#include "utils/utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class TestBernoulliParticleFilter{
public:
	TestBernoulliParticleFilter(string firstFrameFilename, string gtFilename, string detFilename, 
		int num_particles, double lambda, double mu, double epsilon);
	void run();
private:
	int num_particles, num_frames;
	double lambda, mu, epsilon;
	imageGenerator generator;
	double reinit_rate;
	vector<Mat> images;
	vector<VectorXd> detection_weights;
	vector< vector<Rect> > detections;
	vector<string> gt_vec;
};

#endif