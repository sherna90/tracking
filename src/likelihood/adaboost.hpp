// Author: Diego Vergara
#ifndef ADABOOST_H
#define ADABOOST_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include "multinomialnaivebayes.hpp"
#include "weighted_gaussiannaivebayes.hpp"
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <float.h>
#include <math.h>
 
using namespace std;
using namespace Eigen;


class Adaboost
{
public:
    Adaboost();
    Adaboost(string algorithm,int n_estimators, double alpha, double learning_rate);
    void fit(MatrixXd &dX, VectorXi &lX);
    VectorXi predict( MatrixXd &dY);
    MatrixXd get_proba(GaussianNaiveBayes classifier);
    double boost_discrete(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers);
    double boost_real(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers);
    MatrixXd *getdata();
    MatrixXd *gettest();
    VectorXi *getlabels(); 
private:
    int n_estimators, n_data, dim, n_data_test, n_classes;
    string algorithm;
    VectorXd alphas;
    vector<GaussianNaiveBayes> classifiers;
    //std::map<unsigned int,GaussianNaiveBayes, int> classifiers;
    double M_alpha, learning_rate;
    mt19937 generator;
    MatrixXd *dX, *dY;
    VectorXi *lX;
    vector <int> classes;
    bool initialized;
};

#endif