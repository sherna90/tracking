#ifndef HIST_H
#define HIST_H


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"

#define H_BINS 8
#define S_BINS 8

void calc_hist_hsv(cv::Mat& image, cv::Mat& mask, cv::Mat& hist);
void calc_hist_hsv(cv::Mat& image, cv::Mat& hist);
void calc_hog(cv::Mat& image,cv::Mat& hist);
void colorReduce(cv::Mat& image, int div);
#endif