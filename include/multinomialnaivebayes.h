#ifndef MULTINOMIALNAIVEBAYES_H
#define MULTINOMIALNAIVEBAYES_H

#include <stdlib.h>
#include <cmath>
#include "multinomial.h"
#include <map>
#include <iostream>
#include <string>
#include <fstream>


class MultinomialNaiveBayes
{
public:
    MultinomialNaiveBayes();
    MultinomialNaiveBayes(MatrixXd &X, VectorXi &Y);

    void fit();
    VectorXd test(MatrixXd &Xtest);


    std::map<unsigned int, double> getPrior() const;
    void setPrior(const std::map<unsigned int, double> &value);

    MatrixXd *getX();
    void setX(MatrixXd *value);

    VectorXi *getY() ;
    void setY( VectorXi *value);

private:
    MatrixXd *X;
    VectorXi *Y;
    std::map<unsigned int,Multinomial> classes;

    //nuevo método
    std::map<unsigned int,VectorXd> Xc_sufficient;
    //:...............
    std::map<unsigned int,double> Prior;

    bool initialized;


    //-----
};

#endif // MULTINOMIALNAIVEBAYES_H
