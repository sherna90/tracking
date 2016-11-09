#include "hamiltonian_monte_carlo.hpp"


Hamiltonian_MC::Hamiltonian_MC(){
	init = false;
}


Hamiltonian_MC::Hamiltonian_MC(MatrixXd &_X,VectorXd &_Y, double _lambda){
    logistic_regresion = LogisticRegresion(_X,_Y, _lambda);
    init = true;
}

VectorXd Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
	if (init)
	{	
		step_size = _step_size;
		num_step = _num_step;
		/*iterations = _iterations;
		std::default_random_engine generator;
  		std::normal_distribution<double> distribution(0.0,1.0);
  		auto normal = [&] (double) {return distribution(generator);};*/

		//VectorXd initial_x = VectorXd::NullaryExpr(_X.cols(), normal);
		VectorXd initial_x = VectorXd::Zero(_X.cols());
		for (int i = 0; i < _iterations; ++i)
		{
			VectorXd sample = simulation(initial_x);
		}

		return sample
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return 0;
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

  		//VectorXd v0 = VectorXd::NullaryExpr(_initial_x.rows(), normal);
  		VectorXd v0 = VectorXd::Zero(_initial_x.rows());
  		VectorXd x(_initial_x.rows());
  		VectorXd v(_initial_x.rows());

  		leap_Frog(_initial_x, v0, x, v);
		
		double orig = hamiltonian(_initial_x, v0);
		double current = hamiltonian(x, v);
		double p_accept = min(1.0, exp(orig - current));

		if (p_accept > normal(generator))
		{
			return x;
		}
		else{
			return _initial_x;
		}

	}
	else{
		cout << "Error: No initialized function"<< endl;
		return 0;
	}
}

void Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step
	VectorXd v = v0 - 0.5 * step_size * logistic_regresion.Gradient(_x0);
	//Initalize x to be the first step
	VectorXd x = _x0 + step_size * v;

	for (int i = 0; i < num_step; ++i)
	{
		//Compute gradient of the log-posterior with respect to x
		double gradient = logistic_regresion.Gradient(x);
		//Update velocity
		v = v - step_size * gradient;
		//Update x
		x = x + step_size * v;
	}
	//Do a final update of the velocity for a half step
	//return new proposal state
	v = v -0.5 * step_size * logistic_regresion.Gradient(x);

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

	double energy_function = - logistic_regresion.LogPosterior(_position);
	return energy_function(_velocity) + kinetic_energy(_velocity);
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


	