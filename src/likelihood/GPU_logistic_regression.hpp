#ifndef GPU_LOGISTIC_H
#define GPU_LOGISTIC_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "logistic_regression.hpp"

class GPU_LogisticRegression : public LogisticRegression
{
 public:
	void preCompute();
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
 	VectorXd predict(MatrixXd &_X_test, bool prob=false, bool data_processing = true);
 	VectorXd computeGradient();
 	void setGPUFlags(bool _gpu_copy, bool _gpu_free);
 	VectorXd GPU_computeMatrixMul(MatrixXd &m, VectorXd &v);

 private:
 	bool gpu, gpu_copy, gpu_free;
 	void GPU_blasMatrixVectorMul(const float *A, const float *B, float *C, const int m, const int n);
 	void GPU_blasMatrixMatrixMul(const float *A, const float *B, float *C, const int m, const int k, const int n);
 	float *global_matrix_gpu;
};

#endif
