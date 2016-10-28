#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>


using namespace Eigen;

class Multinomial
{
public:
    Multinomial();
    // Multinomial(MatrixXd &counts);
    Multinomial(MatrixXd &counts, double &alpha);
    Multinomial(VectorXd &thetas);
    Multinomial(VectorXd &sufficient,double &alpha);
    double log_likelihood(const VectorXd &test);

    VectorXd getTheta() const;
    void setTheta(const VectorXd &value);
    void addTheta(VectorXd &value, double &alpha);


private:
    VectorXd theta;
    VectorXd sufficient;

};

#endif // MULTINOMIAL_H
