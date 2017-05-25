#ifndef LOGISTIC_H
#define LOGISTIC_H
#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>


using namespace Eigen;
using namespace std;


class LogisticRegression
{
 public:
	LogisticRegression();
	LogisticRegression(MatrixXd &_X,VectorXd &_Y,double lambda=1.0, bool _gpu = false);
	void preCompute();
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd predict(MatrixXd &_X_test, bool prob=false);
 	double logPosterior(VectorXd& _weights);
 	VectorXd gradient(VectorXd &_weights);
    VectorXd computeGradient(MatrixXd &_X, VectorXd &_Y,VectorXd &_W);
    VectorXd computeDataGradient(MatrixXd &_X, VectorXd &_Y, VectorXd &_W);
 	void setWeights(VectorXd &_W);
    void setData(MatrixXd &_X,VectorXd &_Y);
 	VectorXd getWeights();
 	VectorXd GPU_computeMatrixMul(MatrixXd &m, VectorXd &v);


 private:
 	VectorXd weights;
 	MatrixXd *X_train;
 	VectorXd *Y_train;
	VectorXd eta,phi;
 	int rows,dim;
 	double lambda;
 	bool gpu;
 	VectorXd featureMeans;
 	VectorXd sigmoid(VectorXd &_eta);
 	VectorXd logSigmoid(VectorXd &_eta);
 	MatrixXd computeHessian(MatrixXd &_X, VectorXd &_Y, VectorXd &_W);
 	void GPU_blasMatrixVectorMul(const float *A, const float *B, float *C, const int m, const int n);
 	void GPU_blasMatrixMatrixMul(const float *A, const float *B, float *C, const int m, const int k, const int n);
    //MatrixXd computeHessian(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W);
 	double logPrior(VectorXd &_W);
 	double logLikelihood(MatrixXd &_X,VectorXd &_Y);
 	MatrixXd Hessian;
    //MVNGaussian posterior;
};

#endif
