#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "haar.hpp"
#include "hog.hpp"
#include "hist.hpp"
#include "gaussian.hpp"
#include "multinomial.hpp"
#include "LBP.hpp"
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
extern const float  THRESHOLD; 

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace lbp;

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
    Haar haar;
    ~particle_filter();
    particle_filter(int _n_particles);
    particle_filter();
    int time_stamp;
    bool is_initialized();
    void reinitialize();
    void initialize(Mat& current_frame, Rect ground_truth);
    void draw_particles(Mat& image, Scalar color);
    Rect estimate(Mat& image,bool draw);
    void predict();
    void update(Mat& image);
    void smoother(int fixed_lag);
    void update_model(vector<VectorXd> theta_x,vector<VectorXd> theta_y);
    vector<VectorXd> get_dynamic_model();
    vector<VectorXd> get_observation_model();
    float getESS();
    double getMarginalLikelihood();
    void resample();
    vector<Rect> estimates;

protected:
    double marginal_likelihood;
    vector<VectorXd> theta_x;
    vector<VectorXd> theta_y;
    vector<Gaussian> positive_likelihood,negative_likelihood;
    Multinomial color_likelihood,lbp_likelihood,hog_likelihood;
    float ESS;
    bool initialized;
    mt19937 generator;
    Rect reference_roi;
    Size im_size;
    Mat reference_hist;
    normal_distribution<double> position_random_walk,velocity_random_walk,scale_random_walk;
    double eps;
    vector<Rect > sampleBox;
    vector<double > sampleScale;
};

#endif