#ifndef POISSON_H
#define POISSON_H

#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>


using namespace Eigen;

class Poisson
{
public:
    Poisson();
    Poisson(VectorXd &lambda);
    double log_likelihood(const VectorXd &test);

    VectorXd getLambda() const;
    void setLambda(const VectorXd &lambda);
    void addLambda(VectorXd &value);

private:
    VectorXd lambda;
    VectorXd sufficient;
    double sample_size;

};

#endif // POISSON_H
