#ifndef CUDA_HOG_DETECTOR_H
#define CUDA_HOG_DETECTOR_H
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

struct Args {
	bool make_gray = true;
    bool resize_src = true;
    int width, height;
    double scale;
    int nlevels;
    int gr_threshold;
    double hit_threshold;
    bool hit_threshold_auto;
    int win_width;
    int win_stride_width, win_stride_height;
    int block_width;
    int block_stride_width, block_stride_height;
    int cell_width;
    int nbins;
    bool gamma_corr;
    double overlap_threshold;
} ;

class CUDA_HOGDetector
{
public:
	CUDA_HOGDetector(int group_threshold, double hit_threshold);
	vector<Rect> detect(Mat &frame);
	void train(Mat &frame,Rect reference_roi);
	void draw();
	MatrixXd getFeatureValues();
	VectorXd getDetectionWeights();
	Args args;
	MatrixXd compute(Mat &frame);
private:
	int group_threshold;
	double hit_threshold;
	Ptr<cuda::HOG> gpu_hog;
	vector<Rect> detections;
	VectorXd detection_weights;
	Mat frame;
};

#endif
