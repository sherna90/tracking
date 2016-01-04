#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "hist.hpp"
#include "hog.hpp"
#include "dirichlet.hpp"
#include "gaussian.hpp"
#include "multinomial.hpp"
#include "poisson.hpp"
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

extern const float POS_STD; 
extern const float VEL_STD; 
extern const float SCALE_STD; 
extern const float  DT; 
extern const float  SIGMA_COLOR; 
extern const float  SIGMA_SHAPE; 
extern const float  THRESHOLD; 
extern const int  DIRICHLET_LIKELIHOOD; 
extern const int MULTINOMIAL_LIKELIHOOD; 
extern const int POISSON_LIKELIHOOD; 
extern const int LIKELIHOOD;
extern const bool HOG; 
extern const int H_BINS;
extern const int S_BINS;

using namespace cv;
using namespace std;
using namespace Eigen;

typedef struct particle {
    float x; /** current x coordinate */
    float y; /** current y coordinate */
    float width; /** current width coordinate */
    float height; /** current height coordinate */
    float scale; /** current velocity bounding box scale */
    float x_p; /** current x coordinate */
    float y_p; /** current y coordinate */
    float width_p; /** current width coordinate */
    float height_p; /** current height coordinate */
    float scale_p; /** current velocity bounding box scale */
    
} particle;


class particle_filter {
public:
    int n_particles;
    vector<particle> states;
    vector<double>  weights;
    ~particle_filter();
    particle_filter(int _n_particles);
    particle_filter();
    bool is_initialized();
    void reinitialize();
    void initialize(Mat& current_frame, Rect ground_truth);
    void draw_particles(Mat& image);
    Rect estimate(Mat& image,bool draw);
    void predict();
    void update(Mat& image);
    void update_discrete(Mat& image);
    void smoother(int fixed_lag);
    void update_model(VectorXd theta_x,VectorXd theta_y);
    VectorXd get_discrete_model();
    VectorXd get_continuous_model();
    float getESS();
    double getMarginalLikelihood();
    

private:
    double marginal_likelihood;
    VectorXd theta;
    dirichlet polya,polya_hog;
    int time_stamp;
    void resample();
    float ESS;
    bool initialized;
    mt19937 generator;
    Rect reference_roi;
    Size im_size;
    Gaussian color_lilekihood,hog_likelihood;
    Mat reference_hist,reference_hog;
    Multinomial discrete,discrete_hog;
    Poisson poisson;
    normal_distribution<double> position_random_walk,velocity_random_walk,scale_random_walk;
    double eps;

};

#endif