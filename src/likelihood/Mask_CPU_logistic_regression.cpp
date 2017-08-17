#include "Mask_CPU_logistic_regression.hpp"

VectorXd Mask_CPU_LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	for(int i=0;i<n_iter;i++){
		tools.printProgBar(i, n_iter);
		preCompute();
		log_likelihood(i)=-logPosterior();
		VectorXd gradient=this->computeGradient();
		//if( (n_iter/10 % 0)==0) cout << "iteration :   " << i << " | loss : " << log_likelihood(i) << " | Gradient : " <<this->grad_bias  << ","<< gradient.transpose() << endl;
		//cout << "iteration :   " << i << " | Weights : " << this->weights.transpose() << endl;
		this->weights-=alpha*gradient;
		if(this->with_bias) this->bias-=alpha*this->grad_bias;
	}
	cout << endl;
	return log_likelihood;
}

void Mask_CPU_LogisticRegression::setTrainMask(VectorXd _mask){
 	this->train_mask = _mask;
}

void Mask_CPU_LogisticRegression::setTestMask(VectorXd _mask){
 	this->test_mask = _mask;
}

void Mask_CPU_LogisticRegression::preCompute(){
	this->eta = ((this->X_train->array().rowwise() * this->train_mask.array().transpose()).matrix() * this->weights);
	if(this->with_bias) this->eta.noalias()=(this->eta.array()+this->bias).matrix();
	this->phi = sigmoid(this->eta);
}

VectorXd Mask_CPU_LogisticRegression::computeGradient(){
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

VectorXd Mask_CPU_LogisticRegression::predict(MatrixXd &_X_test,bool prob, bool data_processing){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if(this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	VectorXd eta_test = (_X_test.array().rowwise() * this->test_mask.array().transpose()).matrix() *this->weights;
	if(this->with_bias) eta_test.noalias()=(eta_test.array()+this->bias).matrix();
	VectorXd phi_test=sigmoid(eta_test);
	if(!prob){
		phi_test.noalias() = phi_test.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});		
	}
	return phi_test;
}
