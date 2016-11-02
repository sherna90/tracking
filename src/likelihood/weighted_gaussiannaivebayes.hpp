// Author: Diego Vergara
#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <map>
#include <string>
#include <fstream>

using namespace Eigen;
using namespace std;


class GaussianNaiveBayes{
public:
    GaussianNaiveBayes();
    void fit(VectorXd weights);
    VectorXi test(MatrixXd &Xtest);
    MatrixXd get_proba(MatrixXd &Xtest);
    double log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma);
    std::map<unsigned int, double> getPrior() const;
    void setPrior(const std::map<unsigned int, double> &value);
    void setX(MatrixXd *value);
    void setY( VectorXi *value);

private:
    MatrixXd *getX();
    VectorXi *getY() ;
    MatrixXd *X;
    VectorXi *Y;
    std::map<unsigned int,VectorXd> means, sigmas;
    std::map<unsigned int,int> ix;
    std::map<unsigned int,double> Prior, ac_weight;
    bool initialized;
};

#endif // GAUSSIANNAIVEBAYES_H