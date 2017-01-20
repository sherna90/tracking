#ifndef TEST_CLASSIFIERS_H
#define TEST_CLASSIFIERS_H

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

using namespace cv;
using namespace std;
using namespace Eigen;

class TestClassifiers{
public:
	TestClassifiers(string _firstFrame, string _gtFileName);
	void run();
private:
	void initialize();
	void fit();
	void predict();
	imageGenerator imgGenerator;
	vector<Mat> images;
	vector<string> groundtruth;
	unsigned int numTrain, numTest, numFrames;
	mt19937 generator;
	//vector<Rect> positiveBoxes, negativeBoxes;
	MatrixXd xTrain, xTest;
	vector<Rect> positiveTestBoxes, negativeTestBoxes;
	
	C_utils ut;

	Haar haar;
	LocalBinaryPattern local_binary_pattern;

	GaussianNaiveBayes gaussian_naivebayes;
	MultinomialNaiveBayes multinomial_naivebayes;
	Hamiltonian_MC hamiltonian_monte_carlo;

};

#endif
