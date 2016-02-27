#ifndef DISCRETE_PARTICLE_FILTER
#define DISCRETE_PARTICLE_FILTER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "haar.hpp"
#include "hist.hpp"
#include "hog.hpp"
#include "dirichlet.hpp"
#include "gaussian.hpp"
#include "multinomial.hpp"
#include "poisson.hpp"
#include "particle_filter.hpp"
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

using namespace cv;
using namespace std;
using namespace Eigen;


class discrete_particle_filter {
public:
    int n_particles;
    vector<particle> states;
    vector<double>  weights;
    ~discrete_particle_filter();
    discrete_particle_filter(int _n_particles);
    discrete_particle_filter();
    int time_stamp;
    bool is_initialized();
    void reinitialize();
    void initialize(Mat& current_frame, Rect ground_truth);
    void draw_particles(Mat& image, Scalar color);
    Rect estimate(Mat& image,bool draw);
    void predict();
    void update(Mat& image);
    //void update_discrete(Mat& image);
    void smoother(int fixed_lag);
    void update_model(VectorXd theta_x,VectorXd theta_y);
    VectorXd get_dynamic_model();
    VectorXd get_observation_model();
    float getESS();
    double getMarginalLikelihood();
    void resample();
    
protected:
    Multinomial color_likekihood,hog_likelihood;
    double marginal_likelihood;
    VectorXd theta_x,theta_y,theta_hog;
    float ESS;
    bool initialized;
    mt19937 generator;
    Rect reference_roi;
    Size im_size;
    Mat reference_hist,reference_hog;
    normal_distribution<double> position_random_walk,velocity_random_walk,scale_random_walk;
    double eps;
    Haar haar;
    vector<Rect > sampleBox;
};

#endif