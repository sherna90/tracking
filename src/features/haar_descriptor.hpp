#ifndef HAAR_H
#define HAAR_H


#include <opencv2/opencv.hpp>

#include <vector>
#include <Eigen/Dense>

using std::vector;
using namespace cv;

class Haar{
public:
    Haar(void);
	~Haar(void);
	vector<vector<Rect> > features;
	vector<vector<float> > featuresWeight;
private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	int rOuterPositive;

	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	Mat detectFeatureValue;
	RNG rng;

private:
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);
	
}
#endif