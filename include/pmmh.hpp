/**
 * @file pmmh.cpp
 * @brief particle marginal metropolis hastings 
 * @author Sergio Hernandez
 */
#ifndef PMMH
#define PMMH

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "dirichlet.hpp"
#include "gaussian.hpp"
#include "particle_filter.hpp"

//C
#include <stdio.h>
//C++
#include <iostream>
#include <chrono>
#include <queue>
#include <random>
#include <vector>

using namespace cv;
using namespace std;
using namespace Eigen;

class pmmh {
private:
    double marginal_likelihood(VectorXd theta,VectorXd alpha);
    double gamma_prior(VectorXd x,VectorXd a,double b);
    VectorXd discrete_proposal(VectorXd alpha);
    VectorXd continuous_proposal(VectorXd alpha);
    vector<Mat> images;
    Rect reference_roi;
    default_random_engine generator;
    particle_filter* filter;
    VectorXd theta_x,theta_x_prop,theta_y,theta_y_prop,alpha;
    int num_particles,fixed_lag,mcmc_steps;
    Gaussian pos_prior,vel_prior,scale_prior;
    dirichlet color_prior;
    bool initialized;

public:
    pmmh(int num_particles,int fixed_lag,int mcmc_steps);
    void initialize(Mat& image, Rect ground_truth);
    bool is_initialized();
    void reinitialize();
    void update(Mat& image);
    Rect estimate(Mat& image,bool draw);
    ~pmmh();

};

#endif