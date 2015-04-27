#ifndef MULTINOMIALNAIVEBAYES_H
#define MULTINOMIALNAIVEBAYES_H

#include <float.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include "multinomial.h"

using namespace std;

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


    map<int,vector<int> > getXc() const;
    void setXc(const map<int,vector<int> > &value);



    map<int, double> getPrior() const;
    void setPrior(const map<int,double> &value);


private:
    const MatrixXd *X;
    const VectorXi *Y;
    map<int,Multinomial> classes;
    map<int,vector<int> > Xc;
    map<int,int> Tc;
    map<int,double> Prior;


    bool get_classes(),initialized;


    //-----
};

#endif // MULTINOMIALNAIVEBAYES_H
