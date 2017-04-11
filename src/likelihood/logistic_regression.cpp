#include "logistic_regression.hpp"

LogisticRegression::LogisticRegression(){
}

LogisticRegression::LogisticRegression(MatrixXd &_X,VectorXd &_Y,double _lambda){
	srand((unsigned int) time(0));
	this->lambda=_lambda;
 	this->X_train = &_X;
 	this->Y_train = &_Y;
 	VectorXi indices = VectorXi::LinSpaced(this->X_train->rows(), 0, this->X_train->rows());
 	std::random_shuffle(indices.data(), indices.data() + this->X_train->rows());
  	this->X_train->noalias() = indices.asPermutation() * *this->X_train;  
  	this->Y_train->noalias() = indices.asPermutation() * *this->Y_train;
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	this->weights = VectorXd::Random(dim);
	this->eta = VectorXd::Zero(this->rows);
	this->phi = VectorXd::Zero(this->rows);
	this->featureMeans = this->X_train->colwise().mean();
	this->X_train->rowwise()-=this->featureMeans.transpose();
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

void LogisticRegression::preCompute(){
		this->eta = (*X_train*this->weights);
		this->phi = sigmoid(this->eta);
}

VectorXd LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	for(int i=0;i<n_iter;i++){
		//this->eta = (*X_train*this->weights.transpose());
		//this->phi=sigmoid(this->eta);
		preCompute();
		log_likelihood(i)=-logPosterior(this->weights);
		VectorXd Grad=gradient(this->weights);
		this->weights+=alpha*Grad;
		//if(i % (n_iter/10) == 0) cout << "negative log-likelihood : " << log_likelihood(i) << endl;
	}
	return log_likelihood;
}

VectorXd LogisticRegression::computeGradient(MatrixXd &_X, VectorXd &_Y, VectorXd &_W){
	VectorXd E_d=VectorXd::Zero(this->dim);
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
	_X_test.rowwise()-=this->featureMeans.transpose();
	VectorXd eta_test = (_X_test)*this->weights;
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
	double log_likelihood=logLikelihood(*X_train,*Y_train)+logPrior(_weights);
    return log_likelihood;
}

VectorXd LogisticRegression::gradient(VectorXd& _weights){
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
 	VectorXi indices = VectorXi::LinSpaced(this->X_train->rows(), 0, this->X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + this->X_train->rows());
  	this->X_train->noalias() = indices.asPermutation() * *this->X_train;  
  	this->Y_train->noalias() = indices.asPermutation() * *this->Y_train;
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	this->featureMeans = this->X_train->colwise().mean();
}
