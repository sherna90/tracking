#ifndef Mask_GPU_LOGISTIC_H
#define Mask_GPU_LOGISTIC_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "logistic_regression.hpp"

class Mask_GPU_LogisticRegression : public LogisticRegression
{
 public:
	void preCompute();
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd predict(MatrixXd &_X_test, bool prob=false, bool data_processing = true);
 	VectorXd computeGradient();
 	void setTrainMask(VectorXd _mask);
 	void setTestMask(VectorXd _mask);
 	VectorXd GPU_computeMatrixMul(MatrixXd &m, VectorXd &v);

 private:
 	void GPU_blasMatrixVectorMul(const float *A, const float *B, float *C, const int m, const int n);
 	void GPU_blasMatrixMatrixMul(const float *A, const float *B, float *C, const int m, const int k, const int n);
};

#endif
