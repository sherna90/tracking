#ifndef HIST_H
#define HIST_H


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/ocl.hpp>

#define H_BINS 10
#define S_BINS 10

void calc_hist_hsv(cv::Mat& image, cv::Mat& hist);
void calc_hog(cv::Mat& image,cv::Mat& hist);

#endif