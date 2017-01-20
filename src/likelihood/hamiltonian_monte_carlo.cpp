//Author: Diego Vergara
#include "hamiltonian_monte_carlo.hpp"


Hamiltonian_MC::Hamiltonian_MC(){
	init = false;
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

void Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
	if (init)
	{	
		step_size = _step_size;
		num_step = _num_step;
		MatrixXd _weights(_iterations, dim);

		//iterations = _iterations;
		/*std::default_random_engine generator;
  		std::normal_distribution<double> distribution(0.0,1.0);
  		auto normal = [&] (double) {return distribution(generator);};
		VectorXd initial_x = VectorXd::NullaryExpr(dim, normal);*/
		VectorXd initial_x = VectorXd::Random(dim);
		for (int i = 0; i < _iterations; ++i)
		{	

			_weights.row(i) = simulation(initial_x);
			
		}
		weights = _weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

VectorXd Hamiltonian_MC::predict(MatrixXd &_X_test){
	VectorXd predict;
	if (init)
	{	
		VectorXd mean_weights;
		//if(weights.rows()>0) {
			mean_weights = weights.colwise().mean();
		//}
		//else {
		//	mean_weights = VectorXd::Random(dim);
		//}
		logistic_regression.setWeights(mean_weights);
		predict = logistic_regression.predict(_X_test);
		return predict;
		
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

  		VectorXd v0 = VectorXd::Zero(_initial_x.rows());
  		VectorXd x(_initial_x.rows());
  		VectorXd v(_initial_x.rows());
  		leap_Frog(_initial_x, v0, x, v);
		double orig = hamiltonian(_initial_x, v0);
		double current = hamiltonian(x, v);
		double p_accept = min(1.0, exp(orig - current));

		normal_distribution<double> dnormal(0.0,1.0);
		if (p_accept > dnormal(generator))
		{
			return x;
		}
		else{
			return _initial_x;
		}

	}
	else{
		cout << "Error: No initialized function"<< endl;
		return _initial_x;
	}
}

void Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step
	// x(dim);
	// v(dim);
	RowVectorXd rowX(dim);
	rowX << _x0.transpose();
	v = _v0 - 0.5 * step_size * logistic_regression.gradient(rowX);
	//Initalize x to be the first step
	//RowVectorXd x= Map<RowVectorXd>(_x,dim);
	x = _x0 + step_size * v;
	rowX << x.transpose();
	for (int i = 0; i < num_step; ++i)
	{
		//Compute gradient of the log-posterior with respect to x
		VectorXd gradient = logistic_regression.gradient(rowX);
		//Update velocity
		v = v - step_size * gradient;
		//Update x
		x = x + step_size * v;
		rowX << x.transpose();
	}
	//Do a final update of the velocity for a half step
	//return new proposal state
	v = v -0.5 * step_size * logistic_regression.gradient(rowX);

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

	RowVectorXd rowPosition(dim);
	rowPosition << _position.transpose();
	double energy_function = - logistic_regression.logPosterior(rowPosition);
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


void Hamiltonian_MC::setData(MatrixXd &_X,VectorXd &_Y){
	if (init)
	{	
		logistic_regression.setData(_X,_Y);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
       
}
	
	