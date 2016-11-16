//Author: Diego Vergara
#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <fstream>

using namespace Eigen;
using namespace std;


class GaussianNaiveBayes{
public:
    GaussianNaiveBayes();
    GaussianNaiveBayes(MatrixXd &X, VectorXi &Y);
    void fit();
    void partial_fit(MatrixXd &X, VectorXi &Y);
    VectorXi predict(MatrixXd &Xtest);
    MatrixXd get_proba(MatrixXd &Xtest);
    VectorXd predict_proba(MatrixXd &Xtest, int target);
    double log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma);
    double likelihood(VectorXd data, VectorXd mean, VectorXd sigma);
    std::map<unsigned int, double> getPrior() const;
    void setPrior(const std::map<unsigned int, double> &value);
    MatrixXd *getX();
    void setX(MatrixXd *value);
    VectorXi *getY() ;
    void setY( VectorXi *value);

private:
    MatrixXd *X;
    VectorXi *Y;
    std::map<unsigned int,VectorXd> Means, Sigmas;
    std::map<unsigned int,double> Prior;
    bool initialized, one_fit;
    int Rows, Cols;
};

#endif // INCREMENTALGAUSSIANNAIVEBAYES_H