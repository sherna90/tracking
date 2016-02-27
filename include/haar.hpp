#ifndef HAAR_H
#define HAAR_H


#include <opencv2/opencv.hpp>

#include <vector>
#include <Eigen/Dense>

using std::vector;
using namespace cv;

class Haar{
public:
    Haar();
	~Haar();
	vector<vector<Rect> > features;
	vector<vector<float> > featuresWeight;
	Mat sampleFeatureValue;
	int featureNum;
private:
	int featureMinNumRect;
	int featureMaxNumRect;
	Mat imageIntegral;
	Mat detectFeatureValue;
	RNG rng;

private:
	void HaarFeature(Rect& _objectBox, int _numFeature);

public:
	void getFeatureValue(Mat& _frame, vector<Rect>& _sampleBox);
	void init(Mat& _frame, Rect& _objectBox,vector<Rect>& _sampleBox);
	
};
#endif