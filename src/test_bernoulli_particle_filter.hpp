#ifndef TEST_BERNOULLI_PARTICLE_FILTER_H
#define TEST_BERNOULLI_PARTICLE_FILTER_H

#include "models/bernoulli_particle_filter.hpp"
#include "utils/utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestBernoulliParticleFilter{
public:
	TestBernoulliParticleFilter(string _firstFrameFilename, string _gtFilename, int _num_particles);
	void run();
private:
	int num_particles,num_frames;
	imageGenerator generator;
	double reinit_rate;
	vector<Mat> images;
	vector<string> gt_vec;
};

#endif