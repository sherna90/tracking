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
    double gamma_prior(VectorXd x,double a,double b);
    VectorXd continuous_proposal(VectorXd alpha);
    vector<Mat> images;
    Rect reference_roi;
    mt19937 generator;
    particle_filter* filter;
    RowVectorXd theta_x,theta_x_prop,theta_y,theta_y_prop;
    int num_particles,fixed_lag,mcmc_steps;
    
    bool initialized;

public:
    pmmh(int num_particles,int fixed_lag,int mcmc_steps);
    void initialize(Mat& image, Rect ground_truth);
    bool is_initialized();
    void reinitialize();
    void update(Mat& image);
    void draw_particles(Mat& image);
    Rect estimate(Mat& image,bool draw);
    ~pmmh();

};

#endif