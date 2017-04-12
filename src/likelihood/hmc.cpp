//Author: Diego Vergara
#include "hmc.hpp"


Hamiltonian_MC::Hamiltonian_MC(){
	init = false;
	init_2 = false;
}


Hamiltonian_MC::Hamiltonian_MC(MatrixXd &_X, VectorXd &_Y, double _lambda){
	lambda=_lambda;
	X_train = &_X;
 	Y_train = &_Y;
	dim = _X.cols();

    logistic_regression = LogisticRegression(*X_train, *Y_train, _lambda); // Modify Data
    init = true;
    srand((unsigned int) time(0));

}

Hamiltonian_MC::Hamiltonian_MC(MatrixXd &_X, MatrixXd &_data){
	X_train = &_X;
	data = _data;
	dim = _X.cols();

    init_2 = true;
    srand((unsigned int) time(0));

}

VectorXd Hamiltonian_MC::gradient(VectorXd &W, MatrixXd &_data){
	VectorXd grad;
	if (init_2)
	{	
		grad = -this->logistic_regression.gradient(W);
		return grad;
	}
	else{
		return grad;
	}

}
double Hamiltonian_MC::logPosterior(VectorXd &W, MatrixXd &_data){
	double logPost = 0.0;
	if (init_2)
	{
		logPost = -this->logistic_regression.logPosterior(W);
		return logPost;
	}
	else{
		return logPost;
	}

}

VectorXd Hamiltonian_MC::gradient(VectorXd &W){
	VectorXd grad;
	if (init)
	{	
		this->logistic_regression.setWeights(W);
		this->logistic_regression.preCompute();
		grad = -this->logistic_regression.gradient(W);
		return grad;
	}
	else{
		return grad;
	}

}
double Hamiltonian_MC::logPosterior(VectorXd &W){
	double logPost = 0.0;
	if (init)
	{
		this->logistic_regression.setWeights(W);
		this->logistic_regression.preCompute();
		logPost = -this->logistic_regression.logPosterior(W);
		return logPost;
	}
	else{
		return logPost;
	}

}

void Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
	if (init)
	{	
		step_size = _step_size;
		num_step = _num_step;
		weights.resize(_iterations, dim);
		VectorXd initial_x = VectorXd::Random(dim);
		for (int i = 0; i < _iterations; ++i){	
			
			VectorXd initial_v  = VectorXd::Random(dim);
			initial_x = this->simulation(initial_x, initial_v);
			weights.row(i) = initial_x;

		}

		int partition = (int)_iterations*0.5; 
		mean_weights = (weights.block(partition,0 ,weights.rows()-partition, dim)).colwise().mean();
		//mean_weights = weights.colwise().mean();
		//mean_weights = weights.row(weights.rows()-1);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}




VectorXd Hamiltonian_MC::simulation(VectorXd &initial_x, VectorXd &initial_v){
	if (init)
	{
  		VectorXd new_x = initial_x;
  		VectorXd new_v = initial_v;
  		
  		double original = this->hamiltonian(initial_x, initial_v); // old Data
  		
  		this->leap_Frog(new_x, new_v);
  		
		double current = this->hamiltonian(new_x, new_v); // new Data

		// Metopolis-Hasting Correction
		double p_accept = exp(original - current);

		normal_distribution<double> dnormal(0.0,1.0);
		if (p_accept > dnormal(generator))
		{
			return new_x;
		}
		return initial_x;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return initial_x;
	}
}

void Hamiltonian_MC::leap_Frog(VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step

	VectorXd gradient;
	if(init_2){
		gradient = this->gradient(x, data);
	}
	else{
		gradient = this->gradient(x);	
	}
	
	//Initalize x to be the first step
	
	v = v - 0.5 * step_size * gradient;

	for (int i = 0; i < num_step; ++i)
	{
		
		x = x + step_size * v;

		//Compute gradient of the log-posterior with respect to x
		if(init_2){
			gradient = this->gradient(x, data);
		}
		else{
			gradient = this->gradient(x);	
		}
		//Update velocity
		//Update x
		if (i != (num_step-1)) v = v - step_size * gradient;
	}

	v = v - 0.5 * step_size * gradient;

	v = -v;
}

double Hamiltonian_MC::hamiltonian(VectorXd &position, VectorXd &velocity){
	double energy_function = 0.0;
	if(init_2){
	 	energy_function = this->logPosterior(_position, data);
	}
	else{
		energy_function = this->logPosterior(position);
	}

	return energy_function + this->kinetic_energy(velocity);
}

double Hamiltonian_MC::kinetic_energy(VectorXd &velocity){
	/*Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2*/

	return 0.5 * velocity.adjoint()*velocity;
}

VectorXd Hamiltonian_MC::predict(MatrixXd &X_test, bool prob){
	VectorXd predict;
	if (init)
	{	

		this->logistic_regression.setWeights(mean_weights);
		predict = this->logistic_regression.predict(X_test, prob);
		return predict;
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

MatrixXd Hamiltonian_MC::get_weights(){
	MatrixXd predict;
	if (init){	
		return weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

void Hamiltonian_MC::set_weights(VectorXd &_weights){
	if (init){	
		this->mean_weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}