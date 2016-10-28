#ifndef HIST_H
#define HIST_H


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"

extern const int H_BINS;
extern const int S_BINS;

void calc_hist_hsv(cv::Mat& image, cv::Mat& mask, cv::Mat& hist);
void calc_hist_hsv(cv::Mat& image, cv::Mat& hist);
void colorReduce(cv::Mat& image, int div);
#endif