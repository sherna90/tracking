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
#include "../libs/cppoptlib/meta.h"
#include "../libs/cppoptlib/problem.h"
#include "../libs/cppoptlib/solver/bfgssolver.h"
#include "../libs/cppoptlib/solver/lbfgssolver.h"
#include "../libs/cppoptlib/solver/gradientdescentsolver.h"

using namespace Eigen;
using namespace std;

class LogisticRegression
{
 public:
	LogisticRegression();
	LogisticRegression(MatrixXd &_X,VectorXd &_Y,double lambda=1.0);
 	VectorXd Train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd Predict(MatrixXd &_X);
 	double LogPosterior(RowVectorXd& _weights);
 	VectorXd Gradient(RowVectorXd& _weights);
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
 	VectorXd Sigmoid(VectorXd &_eta);
 	VectorXd LogSigmoid(VectorXd &_eta);
 	MatrixXd ComputeHessian(const MatrixXd &_X,  VectorXd &_Y,RowVectorXd &_W);
 	VectorXd ComputeGradient(MatrixXd &_X, VectorXd &_Y,RowVectorXd &_W);
 	double LogPrior(RowVectorXd &_W);
 	double LogLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W);
 	MatrixXd Hessian;
};


template<typename T>
class LogisticRegressionWrapper : public cppoptlib::Problem<T> {
  public:
    using typename cppoptlib::Problem<T>::TVector;
    LogisticRegression *logistic;

    LogisticRegressionWrapper(MatrixXd &X_, VectorXd &y_,double _lambda) {
      logistic=new LogisticRegression(X_,y_,_lambda);
    }

    T value(const TVector &beta) {
        Eigen::RowVectorXd w=beta.transpose();
        return logistic->LogPosterior(w);
    }

    void gradient(const TVector &beta, TVector &grad) {
        Eigen::RowVectorXd w=beta.transpose();
        grad = logistic->Gradient(w);
    }

    int getDim(){
    	return logistic->getWeights().size();
    }
};
#endif