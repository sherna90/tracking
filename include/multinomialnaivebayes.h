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
    MultinomialNaiveBayes(MatrixXd &X, VectorXd &Y);

    void fit();
    VectorXd test(MatrixXd &Xtest);



    std::map<unsigned int, std::vector<unsigned int> > getXc() const;
    void setXc(const std::map<unsigned int, std::vector<unsigned int> > &value);



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
    std::map<unsigned int,std::vector<unsigned int>> Xc;
    std::map<unsigned int,double> Prior;

    bool get_classes(),initialized;


    //-----
};

#endif // MULTINOMIALNAIVEBAYES_H
