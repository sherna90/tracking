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

    logistic_regression = LogisticRegression(_X, _Y, _lambda);
    init = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

Hamiltonian_MC::Hamiltonian_MC(MatrixXd &_X, MatrixXd &_data){
	X_train = &_X;
	data = _data;
	dim = _X.cols();

    init_2 = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

VectorXd Hamiltonian_MC::gradient(VectorXd &weights, MatrixXd &_data){
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
double Hamiltonian_MC::logPosterior(VectorXd &weights, MatrixXd &_data){
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

VectorXd Hamiltonian_MC::gradient(VectorXd &weights){
	VectorXd grad;
	if (init)
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
double Hamiltonian_MC::logPosterior(VectorXd &weights){
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

void Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
	if (init)
	{	
		step_size = _step_size;
		num_step = _num_step;
		MatrixXd _weights(_iterations, dim);

		VectorXd initial_x = VectorXd::Random(dim);

		for (int i = 0; i < _iterations; ++i)
		{	
			
			_weights.row(i) = simulation(initial_x);
			
		}
		weights = _weights;
		mean_weights = weights.colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

VectorXd Hamiltonian_MC::predict(MatrixXd &_X_test){
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

MatrixXd Hamiltonian_MC::predict(){
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


VectorXd Hamiltonian_MC::simulation(VectorXd &_initial_x){
	/*Summary
    Parameters
    ----------
    initial_x : VectorXd
        Initial sample x ~ p
    step_size : double
        Step-size in Hamiltonian simulation
    num_steps : int
        Number of steps to take in Hamiltonian simulation
    log_posterior : object
        Log posterior (unnormalized) for the target distribution
    Returns
    -------
    sample : 
        Sample ~ target distribution
    */

	if (init)
	{

  		VectorXd v0  = VectorXd::Random(_initial_x.rows());
  		VectorXd x(_initial_x.rows());
  		VectorXd v(_initial_x.rows());
  		double orig = hamiltonian(_initial_x, v0); // old Data

  		leap_Frog(_initial_x, v0, x, v);

		double current = hamiltonian(x, v); // new Data

		// Metopolis-Hasting Correction
		double p_accept = exp(orig - current);

		normal_distribution<double> dnormal(0.0,1.0);
		if (p_accept > dnormal(generator))
		{
			_initial_x = x;
		}

		return _initial_x;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return _initial_x;
	}
}

/*void Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step

	VectorXd gradient;
	if(init_2){
		gradient = this->gradient(_x0, data);
	}
	else{
		gradient = this->gradient(x);	
	}
	
	//Initalize x to be the first step
	v= _v0;
	x= _x0;
	for (int i = 0; i < num_step; ++i)
	{
		v = v - 0.5 * step_size * gradient;
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
		v = v - 0.5 * step_size * gradient;
	}

	v = -v;
}*/

void Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step

	VectorXd gradient;
	if(init_2){
		gradient = this->gradient(_x0, data);
	}
	else{
		gradient = this->gradient(x);	
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

	v = -v;
}


double Hamiltonian_MC::hamiltonian(VectorXd &_position, VectorXd &_velocity){
	/*Computes the Hamiltonian of the current position, velocity pair
    H = U(x) + K(v)
    U is the potential energy and is = -log_posterior(x)
    Parameters
    ----------
    position : VectoXd
        Position or state vector x (sample from the target distribution)
    velocity : VectorXd
        Auxiliary velocity variable
    energy_function
        Function from state to position to 'energy'
         = -log_posterior
    Returns
    -------
    hamitonian : double
    */
	
	double energy_function = 0.0;
	if(init_2){
	 	energy_function = this->logPosterior(_position, data);
	}
	else{
		energy_function = -this->logPosterior(_position);
	}

	return energy_function + kinetic_energy(_velocity);
}

double Hamiltonian_MC::kinetic_energy(VectorXd &_velocity){
	/*Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2
    Parameters
    ----------
    velocity : VectorXd
        Vector of current velocity
    Returns
    -------
    kinetic_energy : double
    */

	return 0.5 * _velocity.adjoint()*_velocity;
}