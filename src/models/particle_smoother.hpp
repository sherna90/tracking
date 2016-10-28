#ifndef PARTICLE_SMOOTHER
#define PARTICLE_SMOOTHER


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "../likelihood/hist.hpp"
#include "../likelihood/dirichlet.hpp"
#include "../likelihood/gaussian.hpp"
#include <time.h>
#include <float.h>
#include <vector>
#include <iostream>



using namespace cv;
using namespace std;

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
    int time_stamp;
    float ESS;
    bool initialized;
    RNG rng;
    Rect reference_roi;
    Size im_size;
    VectorXd theta_x,theta_y;
    Gaussian color_lilekihood,hog_likelihood;
    Mat reference_hist,reference_hog;

};

#endif