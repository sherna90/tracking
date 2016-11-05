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
 	RowVectorXd weights;
	LogisticRegression();
	LogisticRegression(MatrixXd &_X,VectorXd &_Y);
 	VectorXd Train(int n_iter,double alpha=0.01,double tol=0.001,double lambda=1.0);
 	VectorXd Predict(MatrixXd &_X);
 	double foo(RowVectorXd& _weights, VectorXd& grad);
 	VectorXd ComputeGradient(MatrixXd &_X, VectorXd &_Y,RowVectorXd &_W,double _lambda);
 	double LogLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W);
 	double LogPrior(RowVectorXd &_W,double lambda);
 
 private:
 	MatrixXd *X_train;
 	VectorXd *Y_train;
 	int rows,dim;
 	VectorXd Sigmoid(VectorXd &_eta);
 	VectorXd LogSigmoid(VectorXd &_eta);
 	MatrixXd ComputeHessian(MatrixXd &_X,  VectorXd &_Y,RowVectorXd &_W,double _lambda);
 	MatrixXd Hessian;
};
