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

VectorXd Hamiltonian_MC::gradient(VectorXd &_W, MatrixXd &_data){
	VectorXd grad;
	if (init_2)
	{	
		grad = -this->logistic_regression.gradient(_W);
		return grad;
	}
	else{
		return grad;
	}

}
double Hamiltonian_MC::logPosterior(VectorXd &_W, MatrixXd &_data){
	double logPost = 0.0;
	if (init_2)
	{
		logPost = -this->logistic_regression.logPosterior(_W);
		return logPost;
	}
	else{
		return logPost;
	}

}

VectorXd Hamiltonian_MC::gradient(VectorXd &_W){
	VectorXd grad;
	if (init)
	{	
		this->logistic_regression.setWeights(_W);
		this->logistic_regression.preCompute();
		grad = -this->logistic_regression.gradient(_W);
		return grad;
	}
	else{
		return grad;
	}

}
double Hamiltonian_MC::logPosterior(VectorXd &_W){
	double logPost = 0.0;
	if (init)
	{
		this->logistic_regression.setWeights(_W);
		this->logistic_regression.preCompute();
		logPost = -this->logistic_regression.logPosterior(_W);
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

		VectorXd old_x = initial_x;
		for (int i = 0; i < _iterations; ++i)
		{	
			weights.row(i) = this->simulation(initial_x, old_x);
			old_x = weights.row(i);
			initial_x = weights.row(i);
			
		}

		int partition = (int)_iterations*0.5; 
		mean_weights = (weights.block(partition,0 ,weights.rows()-partition, dim)).colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

VectorXd Hamiltonian_MC::predict(MatrixXd &_X_test, bool prob){
	VectorXd predict;
	if (init)
	{	

		this->logistic_regression.setWeights(mean_weights);
		predict = this->logistic_regression.predict(_X_test, prob);
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


VectorXd Hamiltonian_MC::simulation(VectorXd &_initial_x, VectorXd &_old_x){

	if (init)
	{

  		VectorXd v0  = VectorXd::Random(dim);
  		VectorXd x(dim);
  		VectorXd v(dim);
  		
  		double orig = this->hamiltonian(_initial_x, v0); // old Data
  		
  		this->leap_Frog(_initial_x, v0, x, v);
  		
		double current = this->hamiltonian(x, v); // new Data

		// Metopolis-Hasting Correction
		double p_accept = exp(orig - current);

		normal_distribution<double> dnormal(0.0,1.0);
		if (p_accept > dnormal(generator))
		{
			return x;
		}

		return _old_x;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return _initial_x;
	}
}

void Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step

	VectorXd gradient;
	if(init_2){
		gradient = this->gradient(_x0, data);
	}
	else{
		gradient = this->gradient(_x0);	
	}
	
	//Initalize x to be the first step
	v= _v0;
	x= _x0;
	
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

	v = -v; // not necesary
}


double Hamiltonian_MC::hamiltonian(VectorXd &_position, VectorXd &_velocity){
	
	double energy_function = 0.0;
	if(init_2){
	 	energy_function = this->logPosterior(_position, data);
	}
	else{
		energy_function = this->logPosterior(_position);
	}

	return energy_function + this->kinetic_energy(_velocity);
}

double Hamiltonian_MC::kinetic_energy(VectorXd &_velocity){
	/*Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2
	*/

	return 0.5 * _velocity.adjoint()*_velocity;
}