#ifndef PARTICLE_SMOOTHER
#define PARTICLE_SMOOTHER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "hist.hpp"
#include "dirichlet.hpp"
#include "gaussian.hpp"
#include "multinomial.hpp"
#include "poisson.hpp"
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
#define DIRICHLET_LIKELIHOOD 0
#define MULTINOMIAL_LIKELIHOOD 1
#define POISSON_LIKELIHOOD 2
#define WITH_HOG true
#define WITHOUT_HOG false

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


class particle_smoother {
public:
    int n_particles;
    vector< vector<particle> > states;
    vector< vector<float> >  weights;
    vector<float>  smoothing_weights;
    particle_smoother(int _n_particles);
    particle_smoother(int _n_particles,VectorXd alpha);
    Rect smoothed_estimate(int fixed_lag);
    void smoother(int fixed_lag);
    void update_model(Mat& previous_frame,Mat& fgmask,Rect& smoothed_estimate);
    float getESS();
    

private:
    dirichlet polya,polya_hog;
    int time_stamp;
    float ESS;
    bool initialized;
    RNG rng;
    Rect reference_roi;
    Size im_size;
    Gaussian color_lilekihood,hog_likelihood;
    Mat reference_hist,reference_hog;
    Multinomial discrete,discrete_hog;
    Poisson poisson;

};

#endif