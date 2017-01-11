//Author: Diego Vergara
#ifndef SG_HMC_H
#define SG_HMC_H
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


class SG_HMC
{
public:
	SG_HMC();
	SG_HMC(MatrixXd &_X,VectorXd &_Y, double _lamda);
	SG_HMC(MatrixXd &_X, MatrixXd &_data);
	void run(double _eta, double _alpha, int _num_step, int _V);
	void partial_run(MatrixXd &_X,VectorXd &_Y);
	void partial_run(MatrixXd &_X, MatrixXd &_data);
	MatrixXd simulation(VectorXd &_initial_x);
	VectorXd predict(MatrixXd &_X_test);
	MatrixXd predict();
	virtual VectorXd stochastic_gradient(VectorXd &weights, MatrixXd &_data);
	virtual double logPosterior(VectorXd &weights, MatrixXd &_data);
	virtual VectorXd stochastic_gradient(VectorXd &weights);
	virtual double logPosterior(VectorXd &weights);

private:
	void leap_Frog(VectorXd &_x0, VectorXd &_v0, MatrixXd &_weights, double momentum, double _sigma);
	//double hamiltonian(VectorXd &_position, VectorXd &_velocity);
	//double kinetic_energy(VectorXd &_velocity);
	bool init, init_2, init_sg;
	double eta, alpha;
	int num_step, dim, V;
 	MatrixXd weights;
 	mt19937 generator;
 	double lambda;
 	MatrixXd *X_train;
 	MatrixXd data;
 	VectorXd *Y_train;
 	VectorXd mean_weights;
 	LogisticRegression logistic_regression;
};

#endif // SG_HMC_H