//Author: Diego Vergara
#ifndef STOCHASTIC_GRADIENT_HAMILTONIAN_MC_H
#define STOCHASTIC_GRADIENT_HAMILTONIAN_MC_H
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


class Stochastic_Gradient_Hamiltonian_MC
{
public:
	Stochastic_Gradient_Hamiltonian_MC();
	Stochastic_Gradient_Hamiltonian_MC(MatrixXd &_X,VectorXd &_Y, double _lamda);
	Stochastic_Gradient_Hamiltonian_MC(MatrixXd &_X, MatrixXd &_data);
	void run(int _iterations, double _m, double _dt, int _num_step, double _C, int _V);
	void partial_run(int new_iterations, MatrixXd &_X,VectorXd &_Y);
	VectorXd simulation(VectorXd &_initial_x);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 0);
	MatrixXd get_weights();
	VectorXd stochastic_gradient(VectorXd &weights);
	double logPosterior(VectorXd &weights);

private:
	VectorXd leap_Frog(VectorXd &_x0, VectorXd &_v0, double D);
	VectorXd random_generator(int dim);
	double random_uniform();
	bool init, init_2, init_sg;
	double m, dt;
	int iterations, num_step, dim, V;
	double C;
 	MatrixXd weights;
 	mt19937 generator;
 	double lambda;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights;
 	LogisticRegression logistic_regression;
};

#endif // STOCHASTIC_GRADIENT_HAMILTONIAN_MC_H