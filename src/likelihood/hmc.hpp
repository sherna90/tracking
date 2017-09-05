//Author: Diego Vergara
#ifndef HAMILTONIAN_MC_H
#define HAMILTONIAN_MC_H
#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>
#include <random>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "multivariate_gaussian.hpp"
#include "../utils/c_utils.hpp"

using namespace Eigen;
using namespace std;

class Hamiltonian_MC
{
public:
	Hamiltonian_MC();
	void acceptace_rate();
	MatrixXd get_weights();
	void set_weights(VectorXd &_weights);
	void set_weightsMatrix(MatrixXd &_weights);
	virtual void run(bool warmup_flag = false,  bool for_predict = false){ cout << "Error, 'run' function, not established" << endl;};
	bool initialized = false;
protected:
	void warmup();
	double avsigmaGauss(double mean, double var);
	VectorXd cumGauss(VectorXd &w, MatrixXd &phi, MatrixXd &Smat);
	VectorXd random_generator(int dim);
	double random_uniform();
	VectorXd random_binomial(int n, VectorXd prob, int dim);
	VectorXd initial_momentum();
	double unif(double step_size);
	bool init_hmc;
	int iterations, warmup_iterations;
	double step_size, path_length;
	int num_step, dim;
	double sampled, accepted;
 	MatrixXd weights;
 	double lambda;
 	MatrixXd inv_cov;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights;
 	VectorXd current_x;
 	MVNGaussian multivariate_gaussian;
 	C_utils tools;
};

#endif // HAMILTONIAN_MC_H