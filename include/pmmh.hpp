/**
 * Particle Marginal Metropolis-Hastings
 * @file pmmh.cpp
 * @brief visual tracking parameter estimation
 * @author Sergio Hernandez
 */
#ifndef PMMH
#define PMMH
#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "../include/dirichlet.hpp"
#include "../include/gaussian.hpp"
#include "../include/multinomial.hpp"
#include "../include/poisson.hpp"
#include "../include/hist.hpp"
#include "../include/particle_filter.hpp"

//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <random>

using namespace cv;
using namespace std;

class PMMH
{
public:
    PMMH(vector<Mat> _images, vector<string> _gt_vect);
    void run(int num_particles,int fixed_lag,int mcmc_steps);

private:
    double marginal_likelihood(int num_particles,int time_step,int fixed_lag,VectorXd theta,VectorXd alpha);
    double gamma_prior(VectorXd x,VectorXd a,double b);
    VectorXd discrete_proposal(VectorXd alpha);
    VectorXd continuous_proposal(VectorXd alpha);
    vector<Mat> images;
    vector<string> gt_vect;
    Rect estimate;
    default_random_engine generator;
};


#endif