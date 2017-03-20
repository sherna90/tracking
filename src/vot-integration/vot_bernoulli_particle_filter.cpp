#include "../models/bernoulli_particle_filter.hpp"

#define VOT_RECTANGLE
#include "vot.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
	if(argc != 9) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
	else{
		int num_particles;
		double lambda, mu, epsilon;
		if(strcmp(argv[1], "-npart") == 0) {
			num_particles = atoi(argv[2]);
		}
		else{
			num_particles = 300;
		}
		if(strcmp(argv[3], "-lambda") == 0) {
			lambda = stod(argv[4]);
		}
		else{
			cerr <<"No lambda given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[5], "-mu") == 0) {
			mu = stod(argv[6]);
		}
		else{
			cerr <<"No mu given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[7], "-epsilon") == 0) {
			epsilon = stod(argv[8]);
		}
		else{
			cerr <<"No epsilon given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		
		BernoulliParticleFilter tracker(num_particles, lambda, mu, epsilon);
		VOT vot;

		Rect initialization;
        initialization << vot.region();
        Mat image = imread(vot.frame());
        tracker.initialize(image, initialization);

        while (!vot.end()) {
          string imagepath = vot.frame();
          if (imagepath.empty()) break;
          Mat image = imread(imagepath);

          tracker.predict();
          tracker.update(image);
          Rect estimate = tracker.estimate(image, true);

          vot.report(estimate);
        }
	}
}
