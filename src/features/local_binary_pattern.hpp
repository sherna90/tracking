#ifndef LOCALBINARYPATTERN_H
#define LOCALBINARYPATTERN_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <climits>
#include <cmath>
#include <cstdio>
#include <complex>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <float.h>

#include "../libs/LBP/LBP.hpp"

using std::vector;
using namespace cv;
using namespace lbp;
using namespace Eigen;

class LocalBinaryPattern{
	public:
		LocalBinaryPattern();
		void getFeatureValue(Mat& _image, vector<Rect> _sampleBox);
		void init(Mat& _image, vector<Rect> _sampleBox, bool _resize = false, bool _complete_image = false, bool _equalized = false);
		MatrixXd sampleFeatureValue;
	private:
		Size initial_size;
		bool initialized;
		int numBlocks;
		int numSupportPoints;
		int rad;
		bool normalizedHist, resize, complete_image, equalized;
		String mapping;
};
#endif
