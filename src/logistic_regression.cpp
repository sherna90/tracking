#include "../include/logistic_regression.hpp"

LogisticRegression::LogisticRegression(MatrixXd _X,VectorXd _Y){
 	X_train = _X;
 	Y_train = _Y;
 	VectorXi indices = VectorXi::LinSpaced(X_train.rows(), 0, X_train.rows());
 	srand((unsigned int) time(0));
 	std::random_shuffle(indices.data(), indices.data() + X_train.rows());
  	X_train.noalias() = indices.asPermutation() * X_train;  
  	Y_train.noalias() = indices.asPermutation() * Y_train; 
 	rows = X_train.rows();
	dim = X_train.cols();
	weights = RowVectorXd::Constant(dim+1,1.0);
	X_train.conservativeResize(NoChange, dim+1);
	VectorXd bias_vec=VectorXd::Constant(rows,1.0);
	X_train.col(dim) = bias_vec;
 }

VectorXd LogisticRegression::ComputeSigmoid(MatrixXd _X, RowVectorXd _W){
	VectorXd phi=VectorXd::Zero(_X.rows());
	VectorXd mu = (-_X*_W.transpose());
	mu.noalias() = mu.unaryExpr([](double elem) // changed type of parameter
	{
		double realmin=numeric_limits<double>::min();
		double maxcut=-log(DBL_EPSILON);
		double mincut=-log(1.0/realmin-1.0);
	    elem=max(elem,mincut);
	    elem=min(elem,maxcut);
	    return elem;
	});
	phi = 1.0 / (1.0 + mu.array().exp()).array();
	return phi;
}

VectorXd LogisticRegression::Train(int n_iter,double alpha,double tol,double lambda){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	MatrixXd H(rows,rows);
	cout << "start training!" << endl;
	for(int i=0;i<n_iter;i++){
		VectorXd Phi=ComputeSigmoid(X_train,weights);
		VectorXd Grad=ComputeGradient(X_train,Y_train,Phi,lambda);
		log_likelihood(i)=LogLikelihood(Y_train,Phi)+LogPrior(lambda);
		weights.noalias()=weights-alpha*Grad.transpose();
	}
	cout << "end training!" << endl;
	VectorXd Phi=ComputeSigmoid(X_train,weights);
	Hessian = ComputeHessian(X_train,Phi,lambda);
	return log_likelihood;
}

VectorXd LogisticRegression::ComputeGradient(MatrixXd _X, MatrixXd _Y,VectorXd _P,double _lambda){
	VectorXd E_d=_X.transpose()*(_P-_Y);
	VectorXd E_w=(_lambda/2.0)*weights.transpose();
	VectorXd grad=E_d+E_w;
	return grad;
}

MatrixXd LogisticRegression::ComputeHessian(MatrixXd _X, VectorXd _P,double _lambda){
	double realmin=numeric_limits<double>::min();
	MatrixXd I=MatrixXd::Identity(dim+1,dim+1);
	MatrixXd H(dim+1,dim+1);
	MatrixXd J(rows,rows);
	J.diagonal() << _P.array()*(1-_P.array()+realmin).array();
	H=_X.transpose()*J;
	H*=_X;
	H+=_lambda*I;
	return H.inverse();
}

VectorXd LogisticRegression::Predict(MatrixXd X_test){
	VectorXd phi=VectorXd::Zero(X_test.rows());
	int n_samples=100;
	MVNGaussian posterior(weights.transpose(),Hessian);
	MatrixXd samples=posterior.sample(n_samples);
	X_test.conservativeResize(NoChange, dim+1);
	VectorXd bias_vec=VectorXd::Constant(X_test.rows(),1.0);
	X_test.col(dim) = bias_vec;
	for(int i=0; i< n_samples;i++){
		phi+=ComputeSigmoid(X_test,samples.row(i));	
	}
	phi.noalias()=(1.0/n_samples)*phi;
	/*phi.noalias() = phi.unaryExpr([](double elem)
	{
	    return (elem > 0.5) ? 1.0 : 0.0;
	});*/
	return phi;
}

double LogisticRegression::LogLikelihood(MatrixXd _Y, VectorXd _P){
	double realmin=numeric_limits<double>::min();
	ArrayXd vec_like=_Y.array()*(_P.array().log());
	vec_like+=(1.0-_Y.array())*( (1.0-_P.array()+realmin).log());
	return vec_like.sum();
}

double LogisticRegression::LogPrior(double _lambda){
	double B=(float)dim+1.0;
	return -(B/2.0)*log(_lambda)+(_lambda/2.0)*weights.array().square().sum();
}