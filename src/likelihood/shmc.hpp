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
	Split_Hamiltonian_MC(MatrixXd &_X,VectorXd &_Y, double _lamda);
	void run(int _iterations, double _step_size, int _num_step);
	void split_run(MatrixXd &_SX, VectorXd &_SY, int _iterations, double _step_size, int _num_step, int _num_splits, int _M);
	//void stochastic_gradient_run(int _iterations, double _step_size, int _num_step, double _m, double _V, double _C);
	//void stochastic_gradient_partial_fit(MatrixXd &_X, VectorXd &_Y, double _lambda, int _iterations);
	VectorXd simulation(VectorXd &_initial_x);
	VectorXd predict(MatrixXd &_X_test, bool prob = true);
	MatrixXd get_weights();
	void fit_map(int _numstart);
	void setData(MatrixXd &_X,VectorXd &_Y);
	virtual VectorXd gradient(VectorXd &weights);
	virtual double logPosterior(VectorXd &weights);
private:
	void leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v);
	//void new_leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v);
	void split_leap_Frog(VectorXd &_x0, VectorXd &_v0, VectorXd &x, VectorXd &v);
	//VectorXd stochastic_gradient_leap_frog();
	double hamiltonian(VectorXd &_position, VectorXd &_velocity);
	double kinetic_energy(VectorXd &_velocity);
	bool init, split, sg;
	//double m, V, C;
	double step_size;
	int num_step, dim, num_splits, M;
 	MatrixXd weights;
 	mt19937 generator;
 	double lambda;
 	MatrixXd *X_train;
 	VectorXd *Y_train;
 	MatrixXd *Split_X_train;
 	VectorXd *Split_Y_train;
 	VectorXd mean_weights;
 	LogisticRegression logistic_regression;
};

#endif // SPLIT_HAMILTONIAN_MC_H