//Author: Diego Vergara
#include "shmc.hpp"


Split_Hamiltonian_MC::Split_Hamiltonian_MC(){
	init = false;
	split = false;
    sg = false;
}


Split_Hamiltonian_MC::Split_Hamiltonian_MC(MatrixXd &_X, VectorXd &_Y, double _lambda){
	lambda=_lambda;
	X_train = &_X;
 	Y_train = &_Y;
	dim = _X.cols();

    logistic_regression = LogisticRegression(_X, _Y, _lambda);
    init = true;
    split = false;
    sg = false;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

VectorXd Split_Hamiltonian_MC::gradient(VectorXd &weights){
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

double Split_Hamiltonian_MC::logPosterior(VectorXd &weights){
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

void Split_Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
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

			_weights.row(i) = this->simulation(initial_x);
			
		}
		weights = _weights;
		//VectorXd mean_weights;
		mean_weights = weights.colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Split_Hamiltonian_MC::split_run(MatrixXd &_SX, VectorXd &_SY, int _iterations, double _step_size, int _num_step, int _num_splits, int _M){
	if (init)
	{	
		M = _M;
		Split_X_train = &_SX;
 		Split_Y_train = &_SY;
		num_splits = _num_splits;
		split = true;
		logistic_regression = LogisticRegression(_SX, _SY, lambda);
		this->run(_iterations, _step_size, _num_step);

	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}


VectorXd Split_Hamiltonian_MC::predict(MatrixXd &_X_test, bool prob){
	VectorXd predict;
	if (init)
	{	
		//VectorXd mean_weights;
		//if(weights.rows()>0) {
		//mean_weights = weights.colwise().mean();
		//}
		//else {
		//	mean_weights = VectorXd::Random(dim);
		//}
		logistic_regression.setWeights(mean_weights);
		predict = logistic_regression.predict(_X_test, prob);
		return predict;
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

MatrixXd Split_Hamiltonian_MC::get_weights(){
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

VectorXd Split_Hamiltonian_MC::simulation(VectorXd &_initial_x){
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

  		VectorXd v0 = VectorXd::Random(_initial_x.rows());
  		VectorXd x(_initial_x.rows());
  		VectorXd v(_initial_x.rows());
  		double orig = this->hamiltonian(_initial_x, v0); // old Data
  		if (!split)
  		{
  			this->leap_Frog(_initial_x, v0, x, v);
  		}
  		else{
  			//VectorXd v0 = VectorXd::Random(_initial_x.rows());
  			VectorXd mean_weights;
  			mean_weights = weights.colwise().mean();
  			this->split_leap_Frog(mean_weights, v0, x, v);
  		}
  		
		double current = this->hamiltonian(x, v); // old Data

		// Metopolis-Hasting Correction
		double p_accept = exp(orig - current);
		//double p_accept = min(1.0, exp(orig - current));

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

/*void Split_Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
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

}*/

/*void Split_Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step
	// x(dim);
	// v(dim);
	RowVectorXd rowX(dim);
	rowX << _x0.transpose();
	VectorXd gradient = logistic_regression.gradient(rowX);
	
	//Initalize x to be the first step
	//RowVectorXd x= Map<RowVectorXd>(_x,dim);
	v= _v0;
	x= _x0;
	for (int i = 0; i < num_step; ++i)
	{
		v = v - 0.5 * step_size * gradient;
		x = x + step_size * v;
		rowX << x.transpose();
		//Compute gradient of the log-posterior with respect to x
		gradient = logistic_regression.gradient(rowX);
		//Update velocity
		//v = v - step_size * gradient;
		//Update x
		//x = x + step_size * v;
		//rowX << x.transpose();
		v = v -0.5 * step_size * gradient;
	}
	//Do a final update of the velocity for a half step
	//return new proposal state
	//v = v -0.5 * step_size * logistic_regression.gradient(rowX);

}*/

void Split_Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step
	VectorXd gradient = this->gradient(_x0);
	
	//Initalize x to be the first step
	v= _v0;
	x= _x0;
	v = v - 0.5 * step_size * gradient;

	for (int i = 0; i < num_step; ++i)
	{
		
		x = x + step_size * v;

		//Compute gradient of the log-posterior with respect to x
		gradient = this->gradient(x);

		//Update velocity
		//Update x
		if (i != (num_step-1)) v = v - step_size * gradient;
		
	}
	//Do a final update of the velocity for a half step
	//return new proposal state
	v = v -0.5 * step_size * gradient;
	v = -v;
	

}

void Split_Hamiltonian_MC::split_leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v){
	//Start by updating the velocity a half-step
	RowVectorXd rowX(dim);
	rowX << _x0.transpose();
	VectorXd gradient_1 = logistic_regression.computeDataGradient(*X_train, *Y_train, rowX); 
	
	//Initalize x to be the first step
	v= _v0;
	x= _x0;
	for (int i = 0; i < num_step; ++i)
	{
		v = v - 0.5 * step_size * gradient_1;
		
		VectorXd gradient_2 = logistic_regression.computeGradient(*Split_X_train, *Split_Y_train, rowX); 

		for (int i = 0; i < num_splits; ++i)
		{
			v = v - (step_size/(2*M)) * gradient_2;
			x = x + (step_size/M) *v;	
			rowX << x.transpose();
			VectorXd gradient_2 = logistic_regression.computeGradient(*Split_X_train, *Split_Y_train, rowX); 
			v = v - (step_size/(2*M)) * gradient_2;
		}

		VectorXd gradient_1 = logistic_regression.computeDataGradient(*X_train, *Y_train, rowX);
		v = v -0.5 * step_size * gradient_1;

	}

}

double Split_Hamiltonian_MC::hamiltonian(VectorXd &_position, VectorXd &_velocity){
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

	double energy_function = - this->logPosterior(_position);
	return energy_function + this->kinetic_energy(_velocity);
}

double Split_Hamiltonian_MC::kinetic_energy(VectorXd &_velocity){
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


void Split_Hamiltonian_MC::fit_map(int _numstart){
	if (init)
	{	
		typedef double T;
    	typedef LogisticRegressionWrapper<T> LogRegWrapper;
    	LogRegWrapper fun(*X_train, *Y_train,lambda);
		MatrixXd _weights(_numstart, dim);
		VectorXd initial_w = VectorXd::Random(dim);
		cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults(); // Create a Criteria class to set the solver's stop conditions
    	cppoptlib::BfgsSolver<LogRegWrapper> solver;
    	solver.setStopCriteria(crit);
		for (int i = 0; i < _numstart; ++i)
		{	
			solver.minimize(fun, initial_w);
			_weights.row(i) = initial_w;
			
		}
		weights = _weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
       
}

void Split_Hamiltonian_MC::setData(MatrixXd &_X,VectorXd &_Y){
	if (init)
	{	
		logistic_regression.setData(_X,_Y);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
       
}
	
	