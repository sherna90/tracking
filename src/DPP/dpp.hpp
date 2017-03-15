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
	/*vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights,VectorXd &penaltyWeights,MatrixXd &featureValues,
	 double alpha, double lambda, double beta, double mu, double epsilon);*/
	vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights,VectorXd &penaltyWeights, MatrixXd &featureValues, 
		VectorXd &qualityTermResults, double lambda, double mu, double epsilon);

private:
	VectorXd get_quality_term(VectorXd &detectionWeights, VectorXd &nPenalty, double lambda);
	MatrixXd get_similarity_term(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon);
	MatrixXd squared_exponential_kernel(MatrixXd X, double nu, double sigma_f);
};

#endif