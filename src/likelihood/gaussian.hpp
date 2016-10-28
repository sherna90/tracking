#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

class Gaussian
{
    public:
        Gaussian(){mean=0;sd=1;};
        Gaussian(double m, double s);
        Gaussian(VectorXd& data);
        double log_likelihood(double test);
        double likelihood(double test);
        double getMean();
        void setMean( double n);
        double getSd();
        void setSd( double n);
    private:
        double mean,sd;
};



#endif 
