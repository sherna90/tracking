//Author: Diego Vergara
#ifndef SPLIT_HAMILTONIAN_MC_H
#define SPLIT_HAMILTONIAN_MC_H
#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "logistic_regression.hpp"

using namespace Eigen;
using namespace std;


class Split_Hamiltonian_MC
{
public:
	Split_Hamiltonian_MC();
	Split_Hamiltonian_MC( MatrixXd &_X, VectorXd &_Y, double _lamda);
	void run(int _iterations, double _step_size, int _num_step);
	void split_run(MatrixXd &_SX, VectorXd &_SY, int new_iterations, double _step_size, int _num_step, int _num_splits, int _M);
	VectorXd dataGradient(MatrixXd &_SX, VectorXd &_SY, VectorXd &W);
	VectorXd gradient(VectorXd &W);
	double logPosterior(VectorXd &W);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 0);
	MatrixXd get_weights();
	void set_weights(VectorXd &_weights);

private:
	VectorXd random_generator(int dim);
	double random_uniform();
	bool init, split;
	double step_size;
	int num_step, dim, iterations;
 	MatrixXd weights;
 	double lambda, old_energy;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights, old_gradient, new_gradient;
 	LogisticRegression logistic_regression;
};

#endif // SPLIT_HAMILTONIAN_MC_H