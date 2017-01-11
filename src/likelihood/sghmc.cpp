//Author: Diego Vergara
#include "sghmc.hpp"

SG_HMC::SG_HMC(){
	init = false;
	init_2 = false;
	init_sg = false;
}


SG_HMC::SG_HMC(MatrixXd &_X, VectorXd &_Y, double _lambda){
	lambda=_lambda;
	X_train = &_X;
 	Y_train = &_Y;
	dim = _X.cols();

    logistic_regression = LogisticRegression(_X, _Y, _lambda);
    init = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

SG_HMC::SG_HMC(MatrixXd &_X, MatrixXd &_data){
	X_train = &_X;
	data = _data;
	dim = _X.cols();

    init_2 = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

VectorXd SG_HMC::stochastic_gradient(VectorXd &weights, MatrixXd &_data){
	VectorXd grad;
	if (init_2)
	{	
		RowVectorXd rowX(dim);
		rowX << weights.transpose();
		grad = logistic_regression.gradient(rowX);
		return grad;
	}
	else{
		return grad;
	}

}

double SG_HMC::logPosterior(VectorXd &weights, MatrixXd &_data){
	double logPost = 0.0;
	if (init_2)
	{
		RowVectorXd rowPosition(dim);
		rowPosition << weights.transpose();
		logPost = -logistic_regression.logPosterior(rowPosition);
		return logPost;
	}
	else{
		return logPost;
	}

}

VectorXd SG_HMC::stochastic_gradient(VectorXd &weights){
	VectorXd grad;
	if (init)
	{	
		RowVectorXd rowX(dim);
		rowX << weights.transpose();
		grad = logistic_regression.gradient(rowX) + VectorXd::Random(dim);
		return grad;
	}
	else{
		return grad;
	}

}

double SG_HMC::logPosterior(VectorXd &weights){
	double logPost = 0.0;
	if (init)
	{
		RowVectorXd rowPosition(dim);
		rowPosition << weights.transpose();
		logPost = -logistic_regression.logPosterior(rowPosition);
		return logPost;
	}
	else{
		return logPost;
	}

}

void SG_HMC::run(double _eta, double _alpha, int _num_step, int _V){
	init_sg = true;
	if (init)
	{	
		eta = _eta;
		alpha = _alpha;
		V = _V;
		num_step = _num_step;

		VectorXd initial_x = VectorXd::Random(dim);
			
		MatrixXd _weights = simulation(initial_x);
			
		weights = _weights;
		mean_weights = weights.colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void SG_HMC::partial_run(MatrixXd &_X, VectorXd &_Y){
	if (init_sg and init)
	{	
		X_train = &_X;
 		Y_train = &_Y;
 		if (_X.cols() == dim)
 		{
			MatrixXd _weights = simulation(mean_weights);
				
			weights = _weights;
			mean_weights = weights.colwise().mean();
 		}
 		else{
 			cout << "Error: Data dimensions are different"<< endl;
 		}

	}
	else{
		cout << "Error: No initialized run SG_HMC or LogisticRegression mode constructor"<< endl;
	}
}

void SG_HMC::partial_run(MatrixXd &_X, MatrixXd &_data){
	if (init_sg and init_2)
	{	
		X_train = &_X;
		data = _data;
 		if (_X.cols() == dim)
 		{
			MatrixXd _weights = simulation(mean_weights);
				
			weights = _weights;
			mean_weights = weights.colwise().mean();
 		}
 		else{
 			cout << "Error: Data dimensions are different"<< endl;
 		}

	}
	else{
		cout << "Error: No initialized run SG_HMC or Generic mode constructor"<< endl;
	}
}


VectorXd SG_HMC::predict(MatrixXd &_X_test){
	VectorXd predict;
	if (init)
	{	
		logistic_regression.setWeights(mean_weights);
		predict = logistic_regression.predict(_X_test, true);
		return predict;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

MatrixXd SG_HMC::predict(){
	MatrixXd predict;
	if (init)
	{	
		return weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}


MatrixXd SG_HMC::simulation(VectorXd &_initial_x){
	MatrixXd _weights(num_step, dim);
	if (init)
	{
		double beta = V*eta*0.5;

		if (beta > alpha)
		{
			cout << "too big eta" << endl;
		}

		double sigma = sqrt(2*eta*(alpha -beta));

  		VectorXd v0  = VectorXd::Random(_initial_x.rows()) * sqrt(eta);

  		double momentum = 1- alpha;

  		leap_Frog(_initial_x, v0, _weights, momentum, sigma);

		return _weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return _weights;
	}
}

void SG_HMC::leap_Frog(VectorXd &_x0, VectorXd &_v0, MatrixXd &_weights, double momentum, double _sigma){

	VectorXd gradient;	
	for (int i = 0; i < num_step; ++i)
	{
		if(init_2){
			gradient = this->stochastic_gradient(_x0, data);
		}
		else{
			gradient = this->stochastic_gradient(_x0);	
		}

		_v0 = _v0 * momentum - gradient * eta + VectorXd::Random(dim) * _sigma;
		_x0 = _x0 + _v0;

		_weights.row(i) = _x0;
	}

}
