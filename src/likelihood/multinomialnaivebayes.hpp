#ifndef MULTINOMIALNAIVEBAYES_H
#define MULTINOMIALNAIVEBAYES_H

#include <stdlib.h>
#include <cmath>
#include "multinomial.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class MultinomialNaiveBayes
{
public:
    MultinomialNaiveBayes();
    MultinomialNaiveBayes(MatrixXd &X, VectorXd &Y);
    void fit(double alpha);
    VectorXd test(MatrixXd &Xtest);
    MatrixXd get_proba(MatrixXd &Xtest);
    std::map<unsigned int, double> getPrior() const;
    void setPrior(const std::map<unsigned int, double> &value);
    MatrixXd *getX();
    void setX(MatrixXd *value);
    VectorXd *getY() ;
    void setY( VectorXd *value);

private:
    MatrixXd *X;
    VectorXd *Y;
    std::map<unsigned int,Multinomial> classes;
    std::map<unsigned int,VectorXd> Xc_sufficient;
    std::map<unsigned int,double> Prior;
    bool initialized;
};

#endif