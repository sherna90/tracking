#include "logistic_regression.hpp"

LogisticRegression::LogisticRegression(){
}

LogisticRegression::LogisticRegression(MatrixXd &_X,VectorXd &_Y,double _lambda){
	srand((unsigned int) time(0));
	this->lambda=_lambda;
 	this->X_train = &_X;
 	this->Y_train = &_Y;
 	/*VectorXi indices = VectorXi::LinSpaced(this->X_train->rows(), 0, this->X_train->rows());
 	std::random_shuffle(indices.data(), indices.data() + this->X_train->rows());
  	this->X_train->noalias() = indices.asPermutation() * *this->X_train;  
  	this->Y_train->noalias() = indices.asPermutation() * *this->Y_train;*/
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	this->weights = VectorXd::Random(dim);
	this->eta = VectorXd::Zero(this->rows);
	this->phi = VectorXd::Zero(this->rows);
	//this->featureMeans = this->X_train->colwise().mean();
	//this->X_train->rowwise()-=this->featureMeans.transpose();
 }

VectorXd LogisticRegression::sigmoid(VectorXd &eta){
	VectorXd phi =eta.unaryExpr([](double elem) // changed type of parameter
	{
	    double p= (elem>0) ? 1.0/(1.0+exp(-elem)) : exp(elem)/(1.0+exp(elem));
	    return p;
	});
	return phi;
}

VectorXd LogisticRegression::logSigmoid(VectorXd &eta){
	VectorXd phi = eta.unaryExpr([](double elem) // changed type of parameter
	{
	   double p= (elem>0) ?  elem-log(1.0+exp(elem)) : -log(1.0+exp(-elem)) ;
	   return p;
	});
	return phi;
}

VectorXd LogisticRegression::GPU_computeMatrixMul(MatrixXd &m, VectorXd &v){
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

	Map<VectorXf> output(h_o, m_rows);

	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_o);

	free(h_m);
	free(h_v);
	free(h_o);

	return output.cast<double>();
}	

void LogisticRegression::GPU_blasMatrixVectorMul(const float *A, const float *B, float *C, const int m, const int n) {
	 int lda=m,ldb=1,ldc=1;
	 const float alf = 1.0;
	 const float bet = 1.0;
	 const float *alpha = &alf;
	 const float *beta = &bet;

	 cublasHandle_t handle;
	 cublasCreate(&handle);

	 cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

	 cublasDestroy(handle);
}

void LogisticRegression::GPU_blasMatrixMatrixMul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
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

void LogisticRegression::preCompute(){
		this->eta = (*X_train*this->weights);
		//this->eta = GPU_computeMatrixMul(*X_train, this->weights);
		this->phi = sigmoid(this->eta);
}

VectorXd LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	for(int i=0;i<n_iter;i++){
		preCompute();
		log_likelihood(i)=-logPosterior(this->weights);
		VectorXd Grad=gradient(this->weights);
		this->weights+=alpha*Grad;
	}
	return log_likelihood;
}

VectorXd LogisticRegression::computeGradient(MatrixXd &_X, VectorXd &_Y, VectorXd &_W){
	VectorXd E_d=VectorXd::Zero(this->dim);
	#pragma omp parallel for schedule(static)
	for(int i=0;i<this->rows;i++){
		double sg=this->phi[i];
		if(_Y[i]>0){
			E_d+=(1.0-sg)*_X.row(i);
		}
		else{
			E_d+=(0.0-sg)*_X.row(i);
		}
	}
	VectorXd E_w=-(-2.0*this->lambda/(double)this->dim)*_W;
	VectorXd grad=(E_d+E_w);
	return grad;
}

VectorXd LogisticRegression::computeDataGradient(MatrixXd &_X, VectorXd &_Y, VectorXd &_W){
	int Dim = _X.cols();
	VectorXd Eta = (_X*_W);
	VectorXd Phi = sigmoid(Eta);
	VectorXd E_d=VectorXd::Zero(Dim);
	#pragma omp parallel for schedule(static)
	for(int i=0;i<this->rows;i++){
		double sg=Phi[i];
		if(_Y[i]>0){
			E_d+=(1.0-sg)*_X.row(i);
		}
		else{
			E_d+=(0.0-sg)*_X.row(i);
		}
	}
	VectorXd E_w=-(-2.0*this->lambda/(double)Dim)*_W;
	VectorXd grad=(E_d+E_w);
	return grad;
}


MatrixXd LogisticRegression::computeHessian(MatrixXd &_X, VectorXd &_Y, VectorXd &_W){
	VectorXd eta = (_X*_W);
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd P=sigmoid(YZ);
	MatrixXd I=MatrixXd::Identity(dim,dim);
	MatrixXd H(dim,dim);
	MatrixXd J(rows,rows);
	J.diagonal() << P.array()*(1-P.array()).array();
	H=_X.transpose()*J;
	H*=_X;
	H+=lambda*I;
	return H.inverse();
}

VectorXd LogisticRegression::predict(MatrixXd &_X_test,bool prob){
	//_X_test.rowwise()-=this->featureMeans.transpose();
	VectorXd eta_test = (_X_test)*this->weights;
	//VectorXd eta_test = GPU_computeMatrixMul(_X_test, this->weights);
	VectorXd phi_test=sigmoid(eta_test);
	if(!prob){
		phi_test.noalias() = phi_test.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});		
	}
	return phi_test;
}

double LogisticRegression::logLikelihood(MatrixXd &_X,VectorXd &_Y){
	double realmin=numeric_limits<double>::min();
	double ll=0.0;
	#pragma omp parallel for schedule(static)
	for(int i=0;i<this->rows;i++){
		double sg=this->phi[i]+realmin;
		if(_Y[i]>0){
			ll+=log(sg);
		}
		else{
			ll+=log(1-sg);
		}
	}
	return ll;
}

double LogisticRegression::logPrior(VectorXd &_W){
	return -(this->lambda/(double)this->dim)*_W.squaredNorm();
}

double LogisticRegression::logPosterior(VectorXd& _weights){
	double log_likelihood=logLikelihood(*X_train,*Y_train);
	double log_prior = logPrior(_weights);
    return log_likelihood + log_prior;
}

VectorXd LogisticRegression::gradient(VectorXd &_weights){
	return computeGradient(*X_train,*Y_train, _weights);
}

void LogisticRegression::setWeights(VectorXd &_W){
	this->weights=_W;
	//Hessian = computeHessian(*X_train,*Y_train,weights);
	//this->posterior = MVNGaussian(weights.transpose(),Hessian);
}

VectorXd LogisticRegression::getWeights(){
	return this->weights;
}

void LogisticRegression::setData(MatrixXd &_X,VectorXd &_Y){
	this->X_train = &_X;
 	this->Y_train = &_Y;
 	/*VectorXi indices = VectorXi::LinSpaced(this->X_train->rows(), 0, this->X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + this->X_train->rows());
  	this->X_train->noalias() = indices.asPermutation() * *this->X_train;  
  	this->Y_train->noalias() = indices.asPermutation() * *this->Y_train;*/
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	//this->featureMeans = this->X_train->colwise().mean();
}
