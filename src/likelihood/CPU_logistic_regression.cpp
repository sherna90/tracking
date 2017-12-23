#include "CPU_logistic_regression.hpp"

VectorXd CPU_LogisticRegression::train(int n_iter,int mini_batch,double alpha,double step_size){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	int num_batches=this->rows/mini_batch;
	VectorXd momemtum=VectorXd::Zero(this->dim);
	cout << "     Epoch     |     Loss        |   Train accuracy  |       Test accuracy  " << endl;
	for(int i=0;i<n_iter;i++){
		this->preCompute(i,mini_batch);
		log_likelihood(i)=-this->logPosterior(i,mini_batch);
		VectorXd gradient=this->computeGradient(i,mini_batch);
		//cout << this->weights.transpose().head(10) << endl;
		//cout << gradient.transpose().head(10) << endl;
		if (i % (n_iter/10) == 0) cout << "              " <<  i/num_batches << "               " << log_likelihood(i)  << "|  " << endl;
		momemtum*=alpha;
		momemtum-=(1.0-alpha)*gradient;
		this->weights+=momemtum*step_size;
		if(this->with_bias) this->bias-=step_size*this->grad_bias;
	}
	return log_likelihood;
}

void CPU_LogisticRegression::preCompute(int iter,int mini_batch){
	int num_batches=this->rows/mini_batch; 
	int idx = iter % num_batches;
	int start = idx * mini_batch;
	int end = (idx + 1) * mini_batch;
	//cout << start << "," << end << endl;
	MatrixXd X_t=this->X_train->block(start,0,mini_batch,this->dim);
	//cout << X_t.rows() << "," << X_t.cols() << endl;
	this->eta = (X_t * this->weights);
	if(this->with_bias) this->eta.noalias()=(this->eta.array()+this->bias).matrix();
	this->phi = this->sigmoid(this->eta);
}

VectorXd CPU_LogisticRegression::computeGradient(int iter,int mini_batch){
	int num_batches=this->rows/mini_batch; 
	int idx = iter % num_batches;
	int start = idx * mini_batch;
	int end = (idx + 1) * mini_batch;
	VectorXd y_slice=this->Y_train->segment(start,mini_batch);
	VectorXd E_d=this->phi-y_slice;
	VectorXd E_w=this->weights*this->lambda;
	MatrixXd X_t=this->X_train->block(start,0,mini_batch,this->dim);
	//VectorXd grad=VectorXd::Zero(this->dim);
	//#pragma omp parallel for schedule(static)
	//for(int d=0;d<this->dim;d++){
	//	grad[d]=X_t.col(d).cwiseProduct(E_d).mean()+E_w[d];
	//}
	//VectorXd grad=(1.0/mini_batch)*X_t.transpose() * E_d + E_w;
	VectorXd grad=X_t.transpose() * E_d + E_w;
	if(this->with_bias) this->grad_bias= (E_d.mean()+this->bias*this->lambda);
	return grad;
}

VectorXd CPU_LogisticRegression::predict(MatrixXd &_X_test,bool prob, bool data_processing){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if (this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	VectorXd eta_test = (_X_test)*this->weights;
	if(this->with_bias) eta_test.noalias()=(eta_test.array()+this->bias).matrix();
	VectorXd phi_test=this->sigmoid(eta_test);
	//cout << phi_test.transpose() << endl;
	if(!prob){
		phi_test.noalias() = phi_test.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});		
	}
	return phi_test;
}
