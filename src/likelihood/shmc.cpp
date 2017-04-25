//Author: Diego Vergara
#include "shmc.hpp"


Split_Hamiltonian_MC::Split_Hamiltonian_MC(){
	this->init = false;
	this->split = false;
}


Split_Hamiltonian_MC::Split_Hamiltonian_MC(MatrixXd &_X, VectorXd &_Y, double _lambda){
	this->lambda=_lambda;
	this->X_train = &_X;
 	this->Y_train = &_Y;
	this->dim = _X.cols();
    this->logistic_regression = LogisticRegression(*X_train, *Y_train, _lambda); // Modify Data
    this->init = true;
    this->split = false;
    this->old_energy = 0.0;
    this->old_gradient = VectorXd::Zero(dim);
    this->new_gradient = VectorXd::Zero(dim);
    this->iterations = 0;

}

VectorXd Split_Hamiltonian_MC::dataGradient(MatrixXd &_SX, VectorXd &_SY, VectorXd &W){
	VectorXd grad;
	if (this->init)
	{	
		grad = -this->logistic_regression.computeDataGradient(_SX, _SY, W);
		return grad;
	}
	else{
		return grad;
	}
}

VectorXd Split_Hamiltonian_MC::gradient(VectorXd &W){
	VectorXd grad;
	if (this->init)
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

double Split_Hamiltonian_MC::logPosterior(VectorXd &W){
	double logPost = 0.0;

	if (this->init){
		this->logistic_regression.setWeights(W);
		this->logistic_regression.preCompute();
		logPost = -this->logistic_regression.logPosterior(W);
		return logPost;
	}
	else{
		return logPost;
	}
}

void Split_Hamiltonian_MC::run(int _iterations, double _step_size, int _num_step){
	if (this->init){	

		this->step_size = _step_size;
		this->num_step = _num_step;
		this->weights.resize(_iterations, this->dim);

		VectorXd x = random_generator(this->dim);
		
		//Hamiltonian
		double Eold = this->logPosterior(x);

		VectorXd p = random_generator(this->dim);

		double lambda = 1.0;
		int n = 0;

		while (n < _iterations){

			VectorXd xold = x;
			VectorXd pold = p;
			double Hold = Eold + 0.5 * p.adjoint()*p;

			if(random_uniform() < 0.5){
				lambda = -1;
			}
			else{
				lambda = 1;
			}

			double epsilon = lambda*this->step_size*(1.0+0.1*random_generator(1)(0));

			p.noalias() = p - 0.5*epsilon*this->gradient(x);
			x.noalias() = x + epsilon*p;

			//Leap Frogs
			for (int i = 0; i < this->num_step; ++i){
				p.noalias() = p - epsilon*this->gradient(x);
				x.noalias() = x + epsilon*p;
			}

			p.noalias() = p - 0.5*epsilon*this->gradient(x);

			//Hamiltonian
			double Enew = this->logPosterior(x);
			p.noalias() = -p;

			double Hnew = Enew + 0.5 * p.adjoint()*p;

			//Metropolis Hasting Correction
			double a = exp(Hold - Hnew);

			if (a > random_uniform()){
				Eold = Enew;
			}
			else{
				x = xold;
				p = pold;
			}
			if (n>=0){
				this->weights.row(n) = x;
			}

			p = random_generator(this->dim);

			n = n+1;

		}

		int partition = (int)_iterations*0.5; 
		this->mean_weights = (this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim)).colwise().mean();
		this->split = true;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Split_Hamiltonian_MC::split_run(MatrixXd &_SX, VectorXd &_SY, int new_iterations, double _step_size, int _num_step, int _num_splits, int _M){
	if (this->init and this->split ){	

		this->step_size = _step_size;
		this->num_step = _num_step;
		int num_splits = _num_splits;
		int old_iterations = this->iterations;
		int M = _M;
		this->iterations += new_iterations;
		this->weights.conservativeResize(this->iterations, NoChange);

		//VectorXd x = random_generator(this->dim);
		VectorXd x = this->mean_weights;
		
		//Hamiltonian
		double Eold = this->logPosterior(x);

		VectorXd p = random_generator(this->dim);

		double lambda = 1.0;
		int n = 0;

		while (n < new_iterations){

			VectorXd xold = x;
			VectorXd pold = p;
			double Hold = Eold + 0.5 * p.adjoint()*p;

			if(random_uniform() < 0.5){
				lambda = -1;
			}
			else{
				lambda = 1;
			}

			double epsilon = lambda*this->step_size*(1.0+0.1*random_generator(1)(0));
			//double epsilon = this->step_size;

			//Leap Frogs

			VectorXd grad = this->gradient(x);

			for (int i = 0; i < this->num_step; ++i){

				p.noalias() = p - 0.5*epsilon*this->dataGradient(_SX, _SY, x);
				
				for (int i = 0; i < num_splits; ++i){

					p.noalias() = p - (epsilon/(2*M)) * grad;
					x.noalias() = x + (epsilon/M) *p;	

					VectorXd grad = this->gradient(x);

					p.noalias() = p - (epsilon/(2*M)) * grad;
				}

				p.noalias() = p -0.5 * epsilon * this->dataGradient(_SX, _SY, x);

			}
			p.noalias() = -p;

			//Hamiltonian
			double Enew = this->logPosterior(x);

			double Hnew = Enew + 0.5 * p.adjoint()*p;

			//Metropolis Hasting Correction
			double a = exp(Hold - Hnew);

			if (a > random_uniform()){
				Eold = Enew;
			}
			else{
				x = xold;
				p = pold;
			}
			if (n>=0){
				this->weights.row(n + old_iterations) = x;
			}

			p = random_generator(this->dim);

			n = n+1;

		}

		int partition = (int)this->iterations*0.5;
		this->mean_weights = (this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim)).colwise().mean();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}


VectorXd Split_Hamiltonian_MC::random_generator(int dimension){
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

double Split_Hamiltonian_MC::random_uniform(){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}


VectorXd Split_Hamiltonian_MC::predict(MatrixXd &X_test, bool prob, int samples){
	VectorXd predict;
	if (this->init){	

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

MatrixXd Split_Hamiltonian_MC::get_weights(){
	MatrixXd predict;

	if (this->init){	
		return this->weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

void Split_Hamiltonian_MC::set_weights(VectorXd &_weights){
	if (this->init){	
		this->mean_weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}