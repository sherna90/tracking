//Author: Diego Vergara
#ifndef APG_LASSO_H
#define APG_LASSO_H
#include <iostream>
#include <iomanip>
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

using namespace Eigen;
using namespace std;


class APG_LASSO
{
public:
	APG_LASSO();
	APG_LASSO(int _iterations, double _beta, double _lamda);
	virtual double objetive_Function(MatrixXd &A, VectorXd &x, VectorXd &b);
	virtual double function(MatrixXd &A, VectorXd &x, VectorXd &b);
	virtual VectorXd gradient(MatrixXd &A, VectorXd &x, VectorXd &b);
	virtual double modelFunction(VectorXd &x, VectorXd &xk, MatrixXd &A, VectorXd &b, double step_lenght);
	VectorXd softThresholding(VectorXd &x, double gamma);
	void fit(MatrixXd &A, VectorXd &b, double _step_lenght);
	VectorXd predict();

private:
	VectorXd matrixDot(MatrixXd &A, VectorXd &x);
	VectorXd sign(VectorXd &x);
	VectorXd vecMax(double value, VectorXd &vec);
	bool init;
	double beta, lambda;
	int iterations;
	VectorXd weights;
};

#endif // APG_LASSO_H