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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "logistic_regression.hpp"
#include "multivariate_gaussian.hpp"

using namespace Eigen;
using namespace std;


class Hamiltonian_MC
{
public:
	Hamiltonian_MC();
	Hamiltonian_MC( MatrixXd &_X, VectorXd &_Y, double _lamda);
	void run(int _iterations, double _step_size, int _num_step);
	VectorXd gradient(VectorXd &W);
	double logPosterior(VectorXd &W);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 0);
	MatrixXd get_weights();
	void set_weights(VectorXd &_weights);

private:
	double avsigmaGauss(double mean, double var);
	VectorXd cumGauss(VectorXd &w, MatrixXd &phi, MatrixXd &Smat);
	VectorXd random_generator(int dim);
	double random_uniform();
	bool init, init_2;
	double step_size;
	int num_step, dim;
 	MatrixXd weights;
 	double lambda, old_energy;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights, old_gradient, new_gradient;
 	LogisticRegression logistic_regression;
};

#endif // HAMILTONIAN_MC_H