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

using namespace Eigen;
using namespace std;


class Hamiltonian_MC
{
public:
	Hamiltonian_MC();
	Hamiltonian_MC(MatrixXd &_X,VectorXd &_Y, double _lamda);
	void run(int _iterations, double _step_size, int _num_step);
	VectorXd simulation(VectorXd &_initial_x);
	VectorXd predict(MatrixXd &_X_test);
	void fit_map(int _numstart);
	void setData(MatrixXd &_X,VectorXd &_Y);
private:
	void leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v);
	double hamiltonian(VectorXd &_position, VectorXd &_velocity);
	double kinetic_energy(VectorXd &_velocity);
	bool init;
	double step_size;
	int num_step, dim;
 	MatrixXd weights;
 	mt19937 generator;
 	double lambda;
 	MatrixXd *X_train;
 	VectorXd *Y_train;
 	LogisticRegression logistic_regression;
};

#endif // HAMILTONIAN_MC_H