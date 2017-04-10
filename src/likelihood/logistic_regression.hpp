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


using namespace Eigen;
using namespace std;

class LogisticRegression
{
 public:
	LogisticRegression();
	LogisticRegression(MatrixXd &_X,VectorXd &_Y,double lambda=1.0);
	void preCompute();
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd predict(MatrixXd &_X_test, bool prob=false);
 	double logPosterior(RowVectorXd& _weights);
 	RowVectorXd gradient(RowVectorXd& _weights);
    RowVectorXd computeGradient(MatrixXd &_X, VectorXd &_Y,RowVectorXd &_W);
 	void setWeights(RowVectorXd &_W);
    void setData(MatrixXd &_X,VectorXd &_Y);
 	RowVectorXd getWeights();


 private:
 	RowVectorXd weights;
 	MatrixXd *X_train;
 	VectorXd *Y_train;
	VectorXd eta,phi;
 	int rows,dim;
 	double lambda;
 	VectorXd featureMeans;
 	VectorXd sigmoid(VectorXd &_eta);
 	VectorXd logSigmoid(VectorXd &_eta);
 	MatrixXd computeHessian(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W);
    //MatrixXd computeHessian(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W);
 	double logPrior(RowVectorXd &_W);
 	double logLikelihood(MatrixXd &_X,VectorXd &_Y);
 	MatrixXd Hessian;
    //MVNGaussian posterior;
};

#endif
