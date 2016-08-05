#ifndef MULTIVARIATE_GAUSSIAN_H
#define MULTIVARIATE_GAUSSIAN_H

#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>

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
    private:
        VectorXd mean;
        MatrixXd cov;
};


#endif // MULTIVARIATE_GAUSSIAN_H