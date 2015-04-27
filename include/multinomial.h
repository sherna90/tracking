#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <float.h>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

class Multinomial
{
public:
    Multinomial();
    Multinomial(MatrixXd &counts);
    Multinomial(MatrixXd &counts, double alpha);
    double log_likelihood(VectorXd test);
    VectorXd getTct() const;
    void setTct(const VectorXd &value);

    VectorXd getTheta() const;
    void setTheta(const VectorXd &value);
private:
    VectorXd theta;
    VectorXd Tct;

};

#endif // MULTINOMIAL_H
