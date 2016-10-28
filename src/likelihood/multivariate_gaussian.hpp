#ifndef MULTIVARIATE_GAUSSIAN_H
#define MULTIVARIATE_GAUSSIAN_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace std;

class MVNGaussian{
    public:
        MVNGaussian(VectorXd _mean, MatrixXd _cov);
        MVNGaussian(MatrixXd &data);
        VectorXd getMean();
        MatrixXd getCov();
        void setMean(VectorXd _mean);
        void setCov(MatrixXd _cov);
        VectorXd log_likelihood(MatrixXd data);
        VectorXd sample();
        MatrixXd sample(int n_samples);
    private:
        VectorXd mean;
        MatrixXd cov;
        mt19937 generator;
        int dim;
};


#endif // MULTIVARIATE_GAUSSIAN_H