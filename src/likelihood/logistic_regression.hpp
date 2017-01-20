#ifndef LOGISTIC_H
#define LOGISTIC_H
#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <algorithm>    // std::max
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "multivariate_gaussian.hpp"


using namespace Eigen;
using namespace std;

class LogisticRegression
{
 public:
	LogisticRegression();
	LogisticRegression(MatrixXd &_X,VectorXd &_Y,double lambda=1.0);
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd predict(MatrixXd &_X,bool prob=true);
 	double logPosterior(RowVectorXd& _weights);
 	VectorXd gradient(RowVectorXd& _weights);
 	void setWeights(VectorXd &_W);
    void setData(MatrixXd &_X,VectorXd &_Y);
 	VectorXd getWeights();


 private:
 	RowVectorXd weights;
 	MatrixXd *X_train;
 	VectorXd *Y_train;
 	int rows,dim;
 	double lambda;
 	VectorXd featureMeans;
 	VectorXd sigmoid(VectorXd &_eta);
 	VectorXd logSigmoid(VectorXd &_eta);
 	MatrixXd computeHessian(const MatrixXd &_X,  VectorXd &_Y,RowVectorXd &_W);
 	VectorXd computeGradient(MatrixXd &_X, VectorXd &_Y,RowVectorXd &_W);
 	double logPrior(RowVectorXd &_W);
 	double logLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W);
 	MatrixXd Hessian;
};

#endif // #define LOGISTIC_H
