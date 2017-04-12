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
#include "likelihood/logistic_regression.hpp"

using namespace Eigen;
using namespace std;


class Hamiltonian_MC
{
public:
	Hamiltonian_MC();
	Hamiltonian_MC( MatrixXd &_X, VectorXd &_Y, double _lamda);
	Hamiltonian_MC( MatrixXd &_X,  MatrixXd &_data);
	void run(int _iterations, double _step_size, int _num_step);
	VectorXd simulation(VectorXd &initial_x, VectorXd &initial_v);
	virtual VectorXd gradient(VectorXd &W, MatrixXd &data);
	virtual double logPosterior(VectorXd &W, MatrixXd &data);
	virtual VectorXd gradient(VectorXd &W);
	virtual double logPosterior(VectorXd &W);
	VectorXd predict(MatrixXd &_X_test, bool prob = false);
	MatrixXd get_weights();
	void set_weights(VectorXd &_weights);

private:
	void leap_Frog(VectorXd &x, VectorXd &v);
	double hamiltonian(VectorXd &position, VectorXd &velocity);
	double kinetic_energy(VectorXd &velocity);
	bool init, init_2;
	double step_size;
	int num_step, dim;
 	MatrixXd weights;
 	mt19937 generator;
 	double lambda;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights;
 	LogisticRegression logistic_regression;
};

#endif // HAMILTONIAN_MC_H