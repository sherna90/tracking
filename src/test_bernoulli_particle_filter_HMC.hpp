#ifndef TEST_BERNOULLI_PARTICLE_FILTER_H
#define TEST_BERNOULLI_PARTICLE_FILTER_H

#include "models/bernoulli_particle_filter_HMC.hpp"
#include "utils/utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestBernoulliParticleFilter{
public:
	TestBernoulliParticleFilter(string firstFrameFilename, string gtFilename, int num_particles, double lambda, double mu, double epsilon);
	void run();
private:
	int num_particles, num_frames;
	double lambda, mu, epsilon;
	imageGenerator generator;
	double reinit_rate;
	vector<Mat> images;
	vector<string> gt_vec;
};

#endif