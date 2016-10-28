// Author: Diego Vergara
#ifndef MB_LBP_H
#define MB_LBP_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
//#include "LBP.hpp"
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <float.h>
#include <math.h>
#include <bitset>
#include <algorithm>
 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

class MultiScaleBlockLBP
{
public:
    MultiScaleBlockLBP();
    MultiScaleBlockLBP(int _p_blocks, int _n_features, int _slider, bool _copy_border, bool _multiscale = false, int _multiscale_slider = 3, int _n_scales = 1);
    void init(Mat& _image, vector<Rect> _sampleBox);
    void getFeatureValue(Mat& _image, vector<Rect> _sampleBox, bool _isPositiveBox);
    int multiScaleBlock_LBP(Mat& d_img, int y, int x);
    void multiScaleBlock_Image(Mat& d_img);
    vector<float> multiScaleBlock_Mapping();
    MatrixXd sampleFeatureValue, negativeFeatureValue;

private:
    double Integrate(Mat& d_img, int r0, int c0, int r1, int c1);
    bool initialized, copy_border, multiscale;
    int initial_p_blocks, p_blocks, n_features, slider, h_size, multiscale_slider, n_scales;
    vector<int> histogram;
};

#endif