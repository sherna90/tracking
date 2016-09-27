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
	LogisticRegression(MatrixXd _X,VectorXd _Y);
 	VectorXd Train(int n_iter,double alpha=0.01,double tol=0.001,double lambda=1.0);
 	VectorXd Predict(MatrixXd _X);
 
 private:
 	MatrixXd X_train;
 	VectorXd Y_train;
 	int rows,dim;
 	VectorXd ComputeSigmoid(MatrixXd _X , RowVectorXd _W);
 	VectorXd ComputeGradient(MatrixXd _X, MatrixXd _Y,VectorXd P,double _lambda);
 	double LogPrior(double lambda);
 	double LogLikelihood(MatrixXd _Y, VectorXd _P);
 	MatrixXd ComputeHessian(MatrixXd _X, VectorXd _P,double _lambda);
 	MatrixXd Hessian;
};