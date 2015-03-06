#ifndef PARTICLE_FILTER

#include "opencv2/core.hpp"
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "hist.h"
#include "dirichlet.h"
#include <time.h>
#include <vector>

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
#define FIXED_LAG 1000

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
    vector<particle> states[FIXED_LAG];
    vector<float>  weights[FIXED_LAG];
    particle_filter(int _n_particles);
    bool is_initialized();
    void initialize(Rect roi,Size im_size);
    void draw_particles(Mat& image);
    Rect estimate(Mat& image,bool draw);
    void predict(Size im_size);
    void update(Mat& image,Mat& reference_hist);
    void update_dirichlet(Mat& image,Mat& reference_hist);
    void update(Mat& image,Mat& reference_hist,Mat& reference_hog);
    float getESS();
    

private:
    int time_stamp;
    void resample();
    float ESS;
    bool initialized;
    RNG rng;
    Rect reference_roi;
};

#endif