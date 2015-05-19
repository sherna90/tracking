#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "hist.hpp"
#include "dirichlet.hpp"
#include "gaussian.hpp"
#include "multinomial.hpp"
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>

#define TRANS_X_STD 0.5
#define TRANS_Y_STD 1.0
#define TRANS_S_STD 0.01
#define POS_STD 1.0
#define VEL_STD 0.01
#define SCALE_STD 0.1
#define DT 1.0
#define SIGMA_COLOR 0.1
#define SIGMA_SHAPE 0.09
#define ALPHA 0.7
#define THRESHOLD 0.7

using namespace cv;
using namespace std;

typedef struct particle {
    float x; /** current x coordinate */
    float y; /** current y coordinate */
    float dx; /** current velocity x coordinate */
    float dy; /** current velocity y coordinate */
    float width; /** current width coordinate */
    float height; /** current height coordinate */
    float scale; /** current velocity bounding box scale */
} particle;


class particle_filter {
public:
    int n_particles;
    vector< vector<particle> > states;
    vector< vector<float> >  weights;
    vector<float>  smoothing_weights;
    particle_filter(int _n_particles);
    particle_filter(int _n_particles,VectorXd alpha);
    bool is_initialized();
    void initialize(Rect roi,Size im_size,Mat& reference_hist,Mat& reference_hog);
    void draw_particles(Mat& image);
    Rect estimate(Mat& image,bool draw);
    Rect smoothed_estimate(int fixed_lag);
    void predict();
    void update(Mat& image,bool hog);
    void update_discrete(Mat& image,bool dirichlet);
    void smoother(int fixed_lag);
    void update_model(Mat& previous_frame,Rect& smoothed_estimate);
    float getESS();
    

private:
    dirichlet polya;
    int time_stamp;
    void resample(bool log_scale);
    float ESS;
    bool initialized;
    RNG rng;
    Rect reference_roi;
    Size im_size;
    Gaussian color_lilekihood,hog_likelihood;
    Multinomial discrete;
    Mat reference_hist,reference_hog;
};

#endif