/**
 * @file pmmh.cpp
 * @brief particle marginal metropolis hastings 
 * @author Sergio Hernandez
 */
#ifndef SMC2
#define SMC2

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "particle_filter.hpp"
#include "utils.hpp"

//C
#include <stdio.h>
#include <iostream>
#include <fstream>
//C++
#include <chrono>
#include <queue>
#include <random>
#include <vector>

using namespace cv;
using namespace std;
using namespace Eigen;

class smc_squared {
private:
    double marginal_likelihood(vector<VectorXd> theta_x,vector<VectorXd> theta_y);
    double igamma_prior(VectorXd x,double a,double b);
    double gamma_prior(VectorXd x,double a,double b);
    VectorXd proposal(VectorXd theta,double step_size);
    vector<Mat> images;
    Rect reference_roi;
    mt19937 generator;
    vector<particle_filter*> filter_bank;
    MatrixXd theta_x_pos,theta_x_scale;
    MatrixXd theta_y_mu,theta_y_sig;
    vector<VectorXd> theta_y_prop,theta_y;
    vector<VectorXd> theta_x_prop,theta_x;
    vector<Rect> estimates;
    int n_particles,m_particles,fixed_lag,mcmc_steps;
    Haar haar;
    bool initialized;

public:
    smc_squared(int num_particles,int fixed_lag,int mcmc_steps);
    void initialize(vector<Mat> _images, Rect ground_truth);
    bool is_initialized();
    void reinitialize(Mat &image, Rect ground_truth);
    void predict();
    void update(Mat& image);
    void run_mcmc();
    void draw_particles(Mat& image);
    Rect estimate(Mat& image,bool draw);
    ~smc_squared();

};

#endif