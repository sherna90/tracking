//Author: Diego Vergara
#include "sghmc.hpp"

Stochastic_Gradient_Hamiltonian_MC::Stochastic_Gradient_Hamiltonian_MC(){
	this->init = false;
	this->init_2 = false;
	this->init_sg = false;
}


Stochastic_Gradient_Hamiltonian_MC::Stochastic_Gradient_Hamiltonian_MC(MatrixXd &_X, VectorXd &_Y, double _lambda){
	this->lambda=_lambda;
	this->X_train = &_X;
 	this->Y_train = &_Y;
	this->dim = _X.cols();
	this->iterations = 0;

    this->logistic_regression = LogisticRegression(_X, _Y, _lambda);
    this->init = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

Stochastic_Gradient_Hamiltonian_MC::Stochastic_Gradient_Hamiltonian_MC(MatrixXd &_X, MatrixXd &_data){
	this->X_train = &_X;
	this->data = _data;
	this->dim = _X.cols();
	this->iterations = 0;

    init_2 = true;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);

}

VectorXd Stochastic_Gradient_Hamiltonian_MC::random_generator(int dimension){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
  	normal_distribution<double> dnormal(0.0,1.0);
	VectorXd random_vector(dimension);

	for (int i = 0; i < dimension; ++i){
		random_vector(i) = dnormal(generator);
	}
	return random_vector;
}

double Stochastic_Gradient_Hamiltonian_MC::random_uniform(){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}


VectorXd Stochastic_Gradient_Hamiltonian_MC::stochastic_gradient(VectorXd &W){
	VectorXd grad;
	if (this->init)
	{	
		this->logistic_regression.setWeights(W);
		this->logistic_regression.preCompute();
		grad = -this->logistic_regression.gradient(W) + random_generator(W.rows());
		return grad;
	}
	else{
		return grad;
	}

}
double Stochastic_Gradient_Hamiltonian_MC::logPosterior(VectorXd &W){
	double logPost = 0.0;
	if (this->init)
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
////////////////////////////////

void Stochastic_Gradient_Hamiltonian_MC::run(int _iterations, double _m, double _dt, int _num_step, double _C, int _V){
	init_sg = true;
	if (this->init){
		
		this->m = _m;
		this->dt = _dt;
		this->V = _V;
		this->C = _C;
		this->num_step = _num_step;
		this->iterations = _iterations;
		this->weights.resize(iterations, this->dim);

		VectorXd initial_x = random_generator(this->dim);
		
		for (int i = 0; i < iterations; ++i){
			initial_x.noalias() = this->simulation(initial_x);
			this->weights.row(i) = initial_x;
		}

		int partition = (int)iterations*0.5; 
		this->mean_weights = (this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim)).colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Stochastic_Gradient_Hamiltonian_MC::partial_run(int new_iterations, MatrixXd &_X, VectorXd &_Y){
	if (this->init_sg and this->init)
	{	
 		if (_X.cols() == this->dim){

			this->X_train = &_X;
 			this->Y_train = &_Y;
 			this->logistic_regression = LogisticRegression(_X, _Y, this->lambda);
 			int old_iterations = this->iterations;
 			this->iterations += new_iterations;
 			this->weights.conservativeResize(this->iterations, NoChange);
 			VectorXd initial_x = this->mean_weights;

 			for (int i = 0; i < new_iterations; ++i){
				initial_x.noalias() = this->simulation(initial_x);
				this->weights.row(i+old_iterations) = initial_x;
			}

			int partition = (int)this->iterations*0.5;
			this->mean_weights = (this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim)).colwise().mean();
 		}
 		else{
 			cout << "Error: Data dimensions are different"<< endl;
 		}

	}
	else{
		cout << "Error: No initialized run Stochastic_Gradient_Hamiltonian_MC or LogisticRegression mode constructor"<< endl;
	}
}

VectorXd Stochastic_Gradient_Hamiltonian_MC::simulation(VectorXd &initial_x){
	VectorXd new_x;
	if (this->init)
	{

  		VectorXd v0  = random_generator(initial_x.rows()) * sqrt(this->m);
  		double B = 0.5*this->V*this->dt;
  		double D = sqrt(2*(this->C-B)*this->dt);

  		new_x = this->leap_Frog(initial_x, v0, D);

		return new_x;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return new_x;
	}
}

VectorXd Stochastic_Gradient_Hamiltonian_MC::leap_Frog(VectorXd &_x0, VectorXd &_v0, double D){
	VectorXd gradient;	
	VectorXd x = _x0;
	VectorXd v = _v0;

	for (int i = 0; i < this->num_step; ++i){
		gradient = this->stochastic_gradient(x);	

		v.noalias() = v - gradient*this->dt - v*this->C*this->dt + random_generator(this->dim)*D;
		x.noalias() = x + (v/this->m)*this->dt;

	}
	return x;
}

VectorXd Stochastic_Gradient_Hamiltonian_MC::predict(MatrixXd &X_test, bool prob, int samples){
	VectorXd predict;
	if (this->init)
	{	
		if (samples == 0){
			this->logistic_regression.setWeights(this->mean_weights);
			predict = this->logistic_regression.predict(X_test, prob);
			return predict;
		}
		else{
			MatrixXd temp_predict(samples, X_test.rows());
			for (int i = 0; i < samples; ++i){
				int randNum = rand()%(weights.rows()-1 + 1) + 0;
				//VectorXd W = this->weights.row(this->weights.rows()-1-i);
				VectorXd W = this->weights.row(randNum);
				this->logistic_regression.setWeights(W);
				temp_predict.row(i) = this->logistic_regression.predict(X_test, prob);
			}
			predict = temp_predict.colwise().mean();
			predict.noalias() = predict.unaryExpr([](double elem){
	    					return (elem > 0.5) ? 1.0 : 0.0;
			});	

			return predict;
		}
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}


MatrixXd Stochastic_Gradient_Hamiltonian_MC::get_weights(){
	MatrixXd predict;
	if (this->init)
	{	
		return this->weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

