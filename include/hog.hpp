#ifndef HOG_H
#define HOG_H


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>
#include <vector>
#include <Eigen/Dense>

void calc_hog(cv::Mat& image,cv::Mat& hist);
void calc_hog(cv::Mat& image,Eigen::VectorXd& hist);

#endif