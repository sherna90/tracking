#include "Mask_GPU_logistic_regression.hpp"

 void Mask_GPU_LogisticRegression::setTrainMask(VectorXd _mask){
 	this->train_mask = _mask;
 }

 void Mask_GPU_LogisticRegression::setTestMask(VectorXd _mask){
 	this->test_mask = _mask;
 }

VectorXd Mask_GPU_LogisticRegression::GPU_computeMatrixMul(MatrixXd &m, VectorXd &v){
	int m_cols = m.cols();
	int m_rows = m.rows();
	int v_cols = v.size();

	float *h_m = (float *)malloc(m_cols * m_rows * sizeof(float));
	float *h_v = (float *)malloc(v_cols * sizeof(float));
	float *h_o = (float *)malloc(m_rows * sizeof(float));

	Map<MatrixXf>(h_m, m_rows, m_cols) = m.cast<float>();
	Map<VectorXf>(h_v, v_cols) = v.cast<float>();

	float *d_m, *d_v, *d_o;

	cudaMalloc((void**)&d_m, m_cols * m_rows * sizeof(float));
	cudaMalloc((void**)&d_v, v_cols * sizeof(float));
	cudaMalloc((void**)&d_o, m_rows * sizeof(float));

	cudaMemcpy(d_m, h_m, m_cols * m_rows * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v, v_cols * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_o, h_o, m_rows * sizeof(float),cudaMemcpyHostToDevice);

	this->GPU_blasMatrixVectorMul(d_m, d_v, d_o, m_rows, v_cols);

	cudaMemcpy(h_o, d_o, m_rows * sizeof(float),cudaMemcpyDeviceToHost);

	Map<VectorXf> outputf(h_o, m_rows);
	VectorXd outputd = outputf.cast<double>();
	
	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_o);

	free(h_m);
	free(h_v);
	free(h_o);

	return outputd;
} 

void Mask_GPU_LogisticRegression::GPU_blasMatrixVectorMul(const float *A, const float *B, float *C, const int m, const int n) {
	 int lda=m,ldb=1,ldc=1;
	 const float alf = 1.0;
	 const float bet = 0.0;
	 const float *alpha = &alf;
	 const float *beta = &bet;

	 cublasHandle_t handle;
	 cublasCreate(&handle);

	 cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

	 cublasDestroy(handle);
}

void Mask_GPU_LogisticRegression::GPU_blasMatrixMatrixMul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	 int lda=m,ldb=k,ldc=m;
	 const float alf = 1.0;
	 const float bet = 1.0;
	 const float *alpha = &alf;
	 const float *beta = &bet;

	 cublasHandle_t handle;
	 cublasCreate(&handle);

	 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cublasDestroy(handle);
}

void Mask_GPU_LogisticRegression::preCompute(){
	MatrixXd tmp = (this->X_train->array().rowwise() * this->train_mask.array().transpose()).matrix();
	this->eta = this->GPU_computeMatrixMul(tmp, this->weights);
	if(this->with_bias) this->eta.noalias()=(this->eta.array()+this->bias).matrix();
	this->phi = sigmoid(this->eta);
}

VectorXd Mask_GPU_LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	for(int i=0;i<n_iter;i++){
		tools.printProgBar(i, n_iter);
		preCompute();
		log_likelihood(i)=-logPosterior();
		VectorXd gradient=this->computeGradient();
		//cout << "iteration :   " << i << " | loss : " << log_likelihood(i) << " | Gradient : " <<this->grad_bias  << ","<< gradient.transpose() << endl;
		//cout << "iteration :   " << i << " | Weights : " << this->weights.transpose() << endl;
		this->weights-=alpha*gradient;
		if(this->with_bias) this->bias-=alpha*this->grad_bias;
	}
	cout << endl;
	return log_likelihood;
}

VectorXd Mask_GPU_LogisticRegression::computeGradient(){
	VectorXd E_d=this->phi-*this->Y_train;
	VectorXd E_w=this->weights/(this->lambda);
	VectorXd grad=VectorXd::Zero(this->dim);
	#pragma omp parallel for schedule(static)
	for(int d=0;d<this->dim;d++){
		grad[d]=(((this->X_train->array().rowwise() * this->train_mask.array().transpose()).col(d)) * E_d.array() ).sum()+E_w[d];
	}
	if(this->with_bias) this->grad_bias= (E_d.sum()+this->bias/this->lambda);
	return grad;
}

VectorXd Mask_GPU_LogisticRegression::predict(MatrixXd &_X_test,bool prob, bool data_processing){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if(this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	MatrixXd tmp = (_X_test.array().rowwise() * this->test_mask.array().transpose()).matrix();
	VectorXd eta_test = GPU_computeMatrixMul(tmp, this->weights);
	if(this->with_bias) eta_test.noalias()=(eta_test.array()+this->bias).matrix();
	VectorXd phi_test=sigmoid(eta_test);
	if(!prob){
		phi_test.noalias() = phi_test.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});		
	}
	return phi_test;
}