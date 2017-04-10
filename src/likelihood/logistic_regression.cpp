#include "logistic_regression.hpp"

LogisticRegression::LogisticRegression(){
}

LogisticRegression::LogisticRegression(MatrixXd &_X,VectorXd &_Y,double _lambda){
	srand((unsigned int) time(0));
	this->lambda=_lambda;
 	this->X_train = &_X;
 	this->Y_train = &_Y;
 	VectorXi indices = VectorXi::LinSpaced(X_train->rows(), 0, X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train->rows());
  	X_train->noalias() = indices.asPermutation() * *X_train;  
  	Y_train->noalias() = indices.asPermutation() * *Y_train;
 	this->rows = X_train->rows();
	this->dim = X_train->cols();
	this->weights = RowVectorXd::Random(dim);
	this->featureMeans = X_train->colwise().mean();
	X_train->rowwise()-=featureMeans.transpose();
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


VectorXd LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	for(int i=0;i<n_iter;i++){
		RowVectorXd Grad=gradient(this->weights);
		log_likelihood(i)=-logPosterior(this->weights);
		this->weights+=alpha*Grad;
		if(i % (n_iter/10) == 0) cout << "negative log-likelihood : " << log_likelihood(i) << endl;
	}
	return log_likelihood;
}


RowVectorXd LogisticRegression::computeGradient(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd Phi=sigmoid(eta);
	RowVectorXd E_d=RowVectorXd::Zero(this->dim);
	for(int i=0;i<this->rows;i++){
		double sg=Phi[i];
		if(_Y[i]>0){
			E_d+=(1.0-sg)*_X.row(i);
		}
		else{
			E_d+=(0.0-sg)*_X.row(i);
		}
	}
	RowVectorXd E_w=-(-2.0*this->lambda/(double)this->dim)*_W;
	RowVectorXd grad=(E_d+E_w);
	return grad;
}

VectorXd LogisticRegression::computeDataGradient(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd Phi=sigmoid(YZ);
	Phi.noalias()=_Y.cwiseProduct((Phi.array()-1).matrix());
	VectorXd E_d=_X.transpose()*Phi;
	return E_d;
}


MatrixXd LogisticRegression::computeHessian(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
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

VectorXd LogisticRegression::predict(MatrixXd &_X,bool log_prob){
	MatrixXd *X_test=&_X;
	X_test->rowwise()-=this->featureMeans.transpose();
	VectorXd phi=VectorXd::Zero(X_test->rows());
	VectorXd eta = (*X_test)*this->weights.transpose();
	if(log_prob){
		phi=logSigmoid(eta);		
	}
	else{
		phi=sigmoid(eta);
		phi.noalias() = phi.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});
	}
	return phi;
}

double LogisticRegression::logLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W){
	double realmin=numeric_limits<double>::min();
	VectorXd eta = (_X*_W.transpose());
	VectorXd Phi=sigmoid(eta);
	double ll=0.0;
	for(int i=0;i<this->rows;i++){
		double sg=Phi[i]+realmin;
		if(_Y[i]>0){
			ll+=log(sg);
		}
		else{
			ll+=log(1-sg);
		}
	}
	return ll;
}

double LogisticRegression::logPrior(RowVectorXd &_W){
	return -(this->lambda/(double)this->dim)*_W.squaredNorm();
}

double LogisticRegression::logPosterior(RowVectorXd& _weights){
	double log_likelihood=logLikelihood(*X_train,*Y_train,_weights)+logPrior(_weights);
    return log_likelihood;
}

RowVectorXd LogisticRegression::gradient(RowVectorXd& _weights){
	return computeGradient(*X_train,*Y_train, _weights);
}

void LogisticRegression::setWeights(VectorXd& _W){
	weights=_W.transpose();
	//Hessian = computeHessian(*X_train,*Y_train,weights);
	//this->posterior = MVNGaussian(weights.transpose(),Hessian);
}

VectorXd LogisticRegression::getWeights(){
	return weights;
}

void LogisticRegression::setData(MatrixXd &_X,VectorXd &_Y){
	X_train = &_X;
 	Y_train = &_Y;
 	/*VectorXi indices = VectorXi::LinSpaced(X_train->rows(), 0, X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train->rows());
  	X_train->noalias() = indices.asPermutation() * *X_train;  
  	Y_train->noalias() = indices.asPermutation() * *Y_train; */
 	rows = X_train->rows();
	dim = X_train->cols();
	//featureMeans = X_train->colwise().mean();
}