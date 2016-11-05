#include "logistic_regression.hpp"

LogisticRegression::LogisticRegression(){
}

LogisticRegression::LogisticRegression(MatrixXd &_X,VectorXd &_Y){
 	X_train = &_X;
 	Y_train = &_Y;
 	VectorXi indices = VectorXi::LinSpaced(X_train->rows(), 0, X_train->rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train->rows());
  	X_train->noalias() = indices.asPermutation() * *X_train;  
  	Y_train->noalias() = indices.asPermutation() * *Y_train; 
 	rows = X_train->rows();
	dim = X_train->cols();
	weights = RowVectorXd::Zero(dim+1);
	X_train->conservativeResize(NoChange, dim+1);
	VectorXd bias_vec=VectorXd::Constant(rows,1.0);
	X_train->col(dim) = bias_vec;
 }

VectorXd LogisticRegression::Sigmoid(VectorXd &eta){
	VectorXd phi =eta.unaryExpr([](double elem) // changed type of parameter
	{
		double realmin=numeric_limits<double>::min();
		double maxcut=-log(DBL_EPSILON);
		double mincut=-log(1.0/realmin-1.0);
	    elem=max(elem,mincut);
	    elem=min(elem,maxcut);
	    double p= (elem>0) ? 1.0/(1.0+exp(-elem)) : exp(elem)/(1.0+exp(elem));
	    return p;
	});
	return phi;
}

VectorXd LogisticRegression::LogSigmoid(VectorXd &eta){
	VectorXd phi = eta.unaryExpr([](double elem) // changed type of parameter
	{
		double realmin=numeric_limits<double>::min();
		double maxcut=-log(DBL_EPSILON);
		double mincut=-log(1.0/realmin-1.0);
	    elem=max(elem,mincut);
	    elem=min(elem,maxcut);
	    double p= (elem>0) ? -log(1.0+exp(-elem)) : elem-log(1.0+exp(elem));
	    return p;
	});
	return phi;
}

VectorXd LogisticRegression::Train(int n_iter,double alpha,double tol,double lambda){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	MatrixXd H(rows,rows);
	cout << "start training!" << endl;
	for(int i=0;i<n_iter;i++){
		VectorXd Grad=ComputeGradient(*X_train,*Y_train,weights,lambda);
		log_likelihood(i)=-LogLikelihood(*X_train,*Y_train,weights)-LogPrior(weights,lambda);
		//cout << i << ", w:" << weights <<  ", ll:" << log_likelihood(i)  <<  ", g:" << Grad.transpose() <<endl;
		weights.noalias()=weights-alpha*Grad.transpose();
	}
	cout << "end training!" << endl;
	Hessian = ComputeHessian(*X_train,*Y_train,weights,lambda);
	return log_likelihood;
}

VectorXd LogisticRegression::ComputeGradient(MatrixXd &_X, VectorXd &_Y, RowVectorXd &_W,double _lambda){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd Phi=Sigmoid(YZ);
	Phi.noalias()=_Y.cwiseProduct((Phi.array()-1).matrix());
	VectorXd E_d=_X.transpose()*Phi;
	VectorXd E_w=(_lambda)*weights.transpose();
	VectorXd grad=(E_d+E_w);
	return grad;
}

MatrixXd LogisticRegression::ComputeHessian(MatrixXd &_X,  VectorXd &_Y,RowVectorXd &_W,double _lambda){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd P=Sigmoid(YZ);
	MatrixXd I=MatrixXd::Identity(dim+1,dim+1);
	MatrixXd H(dim+1,dim+1);
	MatrixXd J(rows,rows);
	J.diagonal() << P.array()*(1-P.array()).array();
	H=_X.transpose()*J;
	H*=_X;
	H+=_lambda*I;
	return H.inverse();
}

VectorXd LogisticRegression::Predict(MatrixXd &_X){
	MatrixXd *X_test=&_X;
	X_test->conservativeResize(NoChange, dim+1);
	VectorXd bias_vec=VectorXd::Constant(X_test->rows(),1.0);
	X_test->col(dim) = bias_vec;
	VectorXd phi=VectorXd::Zero(X_test->rows());
	int n_samples=100;
	MVNGaussian posterior(weights.transpose(),Hessian);
	MatrixXd samples=posterior.sample(n_samples);
	for(int i=0; i< n_samples;i++){
		RowVectorXd sample_weight=samples.row(i);
		VectorXd eta = (*X_test)*sample_weight.transpose();
		phi+=(1.0/n_samples)*Sigmoid(eta);	
	}
	/*phi.noalias() = phi.unaryExpr([](double elem)
	{
	    return (elem > 0.5) ? 1.0 : -1.0;
	});
	cout << Hessian << endl;*/
	return phi;
}

double LogisticRegression::LogLikelihood(MatrixXd &_X,VectorXd &_Y,RowVectorXd &_W){
	VectorXd eta = (_X*_W.transpose());
	VectorXd YZ=_Y.cwiseProduct(eta);
	VectorXd logPhi=LogSigmoid(YZ);
	return logPhi.sum();
}

double LogisticRegression::LogPrior(RowVectorXd &_W,double _lambda){
	return -(_lambda/2.0)*_W.squaredNorm();
}

double LogisticRegression::foo(RowVectorXd& _weights, VectorXd& grad){
	grad.noalias()=ComputeGradient(*X_train,*Y_train,_weights,0.1);
	double log_likelihood=-LogLikelihood(*X_train,*Y_train,_weights)-LogPrior(_weights,0.1);
    return log_likelihood;
}