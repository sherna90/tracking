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
    VectorXi test(MatrixXd &Xtest);

    const MatrixXd *getX() const;
    void setX(const MatrixXd *value);

    const VectorXi *getY() const;
    void setY(const VectorXi *value);


    std::map<long, std::vector<long> > getXc() const;
    void setXc(const std::map<long, std::vector<long> > &value);



    std::map<long, double> getPrior() const;
    void setPrior(const std::map<long, double> &value);


private:
    const MatrixXd *X;
    const VectorXi *Y;
    std::map<long,Multinomial> classes;
    std::map<long,std::vector<long>> Xc;
    std::map<long,double> Prior;

    bool get_classes(),initialized;


    //-----
};

#endif // MULTINOMIALNAIVEBAYES_H
