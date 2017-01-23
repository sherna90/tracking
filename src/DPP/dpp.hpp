#ifndef DPP_H
#define DPP_H

#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

class DPP
{
public:
	DPP();
	vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues, double alpha, double lambda, double beta, double mu, double epsilon);

private:
	VectorXd getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty, double alpha, double beta);
	MatrixXd getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon);
};

#endif