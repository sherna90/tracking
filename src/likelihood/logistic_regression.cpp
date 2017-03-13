#include "logistic_regression.hpp"

LogisticRegression::LogisticRegression(){
}

LogisticRegression::LogisticRegression(MatrixXd &_X,VectorXd &_Y,double _lambda){
	lambda=_lambda;
 	X_train = &_X;
 	Y_train = &_Y;
 	VectorXi indices = VectorXi::LinSpaced(X_train->rows(), 0, X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train->rows());
  	X_train->noalias() = indices.asPermutation() * *X_train;  
  	Y_train->noalias() = indices.asPermutation() * *Y_train; 
 	rows = X_train->rows();
	dim = X_train->cols();
	weights = RowVectorXd::Random(dim);
	featureMeans = X_train->colwise().mean();
	X_train->rowwise()-=featureMeans.transpose();
	/*X_train->conservativeResize(NoChange, dim+1);
	VectorXd bias_vec=VectorXd::Constant(rows,1.0);
	X_train->col(dim) = bias_vec;*/
 }

VectorXd LogisticRegression::sigmoid(VectorXd &eta){
	VectorXd phi =eta.unaryExpr([](double elem) // changed type of parameter
	{
		double maxcut=log(std::numeric_limits<float>::max_exponent);
		double mincut=log(std::numeric_limits<float>::min_exponent);
	    elem=max(elem,mincut);
	    elem=min(elem,maxcut);
	    double p= (elem>0) ? 1.0/(1.0+exp(-elem)) : exp(elem)/(1.0+exp(elem));
	    return p;
	});
	return phi;
}

VectorXd LogisticRegression::logSigmoid(VectorXd &eta){
	VectorXd phi = eta.unaryExpr([](double elem) // changed type of parameter
	{
		//double realmin=numeric_limits<double>::min();
		double maxcut=log(std::numeric_limits<float>::max_exponent);
		double mincut=log(std::numeric_limits<float>::min_exponent);
	    elem=max(elem,mincut);
	    elem=min(elem,maxcut);
	    double p= (elem>0) ? -log(1.0+exp(-elem)) : elem-log(1.0+exp(elem));
	    return p;
	});
	return phi;
}


VectorXd LogisticRegression::train(int n_iter,double alpha,double tol){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	MatrixXd H(rows,rows);
	for(int i=0;i<n_iter;i++){
		VectorXd Grad=gradient(weights);
		log_likelihood(i)=logPosterior(weights);
		//cout << i << ", ll:" << log_likelihood(i)  <<endl;
		weights.noalias()=weights-alpha*Grad.transpose();
	}
	//cout << "end training!" << endl;
	Hessian = computeHessian(*X_train,*Y_train,weights);
	return log_likelihood;
}


VectorXd LogisticRegression::computeGradient(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd Phi=sigmoid(YZ);
	Phi.noalias()=_Y.cwiseProduct((Phi.array()-1).matrix());
	VectorXd E_d=_X.transpose()*Phi;
	VectorXd E_w=(lambda)*weights.transpose();
	VectorXd grad=(E_d+E_w);
	return grad;
}

MatrixXd LogisticRegression::computeHessian(const MatrixXd &_X,  VectorXd &_Y,RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	MatrixXd I=MatrixXd::Identity(dim,dim);
	VectorXd P=sigmoid(YZ);
	MatrixXd H=MatrixXd::Zero(dim,dim);
	MatrixXd J=MatrixXd::Zero(rows,rows);
	J.diagonal() << P.array()*(1-P.array()).array();
	//cout << "data " << _Y.rows() << "," << _Y.cols() << "," << _X.rows() << "," << _X.cols() << endl;
	MatrixXd H_temp=_X.transpose();
	//H_temp *= J;
	//H.noalias()=H_temp*_X;
	H+=lambda*I;
	return H.inverse();
}

VectorXd LogisticRegression::predict(MatrixXd &_X,bool prob){
	MVNGaussian posterior(weights.transpose(),Hessian);
	//cout << "data " << Y_train->rows() << "," << Y_train->cols() << "," << X_train->rows() << "," << X_train->cols() << endl;
	MatrixXd *X_test=&_X;
	//X_test->rowwise()-=featureMeans.transpose();
	VectorXd phi=VectorXd::Zero(X_test->rows());
	VectorXd eta = (*X_test)*weights.transpose();
	if(prob){
		phi=logSigmoid(eta);		
	}
	else{
		phi=sigmoid(eta);
		//phi.noalias() = phi.unaryExpr([](double elem){
	    //	return (elem > 0.5) ? 1.0 : -1.0;
		//});
	}
	/*int n_samples=100;
	for(int i=0; i< n_samples;i++){
		VectorXd sample_weight=posterior.sample();
		VectorXd eta = *X_test*sample_weight;
		phi+=(1.0/n_samples)*sigmoid(eta);	
	}*/
	return phi;
}

double LogisticRegression::logLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd logPhi=logSigmoid(YZ);
	return logPhi.sum();
}

double LogisticRegression::logPrior(RowVectorXd &_W){
	return -(lambda/2.0)*_W.squaredNorm();
}

double LogisticRegression::logPosterior(RowVectorXd& _weights){
	double log_likelihood=-logLikelihood(*X_train,*Y_train,_weights)-logPrior(_weights);
    return log_likelihood;
}

VectorXd LogisticRegression::gradient(RowVectorXd& _weights){
	return computeGradient(*X_train,*Y_train, _weights);
}

void LogisticRegression::setWeights(VectorXd& _W){
	weights=_W.transpose();
}

VectorXd LogisticRegression::getWeights(){
	return weights;
}

void LogisticRegression::setData(MatrixXd &_X,VectorXd &_Y){
	X_train = &_X;
 	Y_train = &_Y;
 	VectorXi indices = VectorXi::LinSpaced(X_train->rows(), 0, X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train->rows());
  	X_train->noalias() = indices.asPermutation() * *X_train;  
  	Y_train->noalias() = indices.asPermutation() * *Y_train; 
 	rows = X_train->rows();
	dim = X_train->cols();
	featureMeans = X_train->colwise().mean();
}