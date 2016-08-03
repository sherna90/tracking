#include "../include/haar.hpp"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

//------------------------------------------------
Haar::Haar(){
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
}

Haar::~Haar(){
}	

void Haar::HaarFeature(Rect& _objectBox, int _numFeature){
	features = vector<vector<Rect> >(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float> >(_numFeature, vector<float>());
	
	int numRect,dim;
	Rect rectTemp,halfRect;
	float weightTemp;
	for (int i=0; i<_numFeature; i++)
	{
		//cout << "Feature :" << i << ";" ; 
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
		rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
		rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
		rectTemp.width = cvCeil(rng.uniform(8.0, (double)(_objectBox.width - rectTemp.x - 2)));
		rectTemp.height = cvCeil(rng.uniform(8.0, (double)(_objectBox.height - rectTemp.y - 2)));
		dim = rng.uniform(0, 2);
		//cout <<dim <<",";
		for (int j=0; j<numRect; j++)
		{
			if(dim==0){
				halfRect.width=cvCeil(rectTemp.width/numRect);
				halfRect.x=cvFloor(rectTemp.x+j*halfRect.width);
				halfRect.y=rectTemp.y;
				halfRect.height=rectTemp.height;
			}
			else if(dim==1){
				halfRect.width=rectTemp.width;
				halfRect.x=rectTemp.x;
				halfRect.height=cvCeil(rectTemp.height/numRect);
				halfRect.y=cvFloor(rectTemp.y+j*halfRect.height);
			}
			features[i].push_back(halfRect);
			//weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			weightTemp = (j % 2 == 0) ? -1.0 : 1.0;
			featuresWeight[i].push_back(weightTemp);
   			//cout << weightTemp <<"," << halfRect  ;
		}
		//cout << endl;
	}
}

void Haar::getFeatureValue(Mat& _frame, vector<Rect>& _sampleBox, vector<double> _sampleScale)
{
	integral(_frame, imageIntegral, CV_32F);
	int sampleBoxSize = _sampleBox.size();
	sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[i].size(); k++)
			{
				xMin = _sampleBox[j].x + cvRound(features[i][k].x*_sampleScale[j]);
				xMax = _sampleBox[j].x + cvRound(features[i][k].x*_sampleScale[j] + features[i][k].width*_sampleScale[j]);
				yMin = _sampleBox[j].y + cvRound(features[i][k].y*_sampleScale[j]);
				yMax = _sampleBox[j].y + cvRound(features[i][k].y*_sampleScale[j] + features[i][k].height*_sampleScale[j]);
				//cout << xMin << ","<< xMax << ","<< yMin << ","<< yMax<< endl;
				//cout << _frame.size() << endl;
				if(xMax < _frame.cols && yMax < _frame.rows){
					tempValue += featuresWeight[i][k] *
						(imageIntegral.at<float>(yMin, xMin) +
						imageIntegral.at<float>(yMax, xMax) -
						imageIntegral.at<float>(yMin, xMax) -
						imageIntegral.at<float>(yMax, xMin));
				}
			}
			sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
}

void Haar::init(Mat& _frame, Rect& _objectBox,vector<Rect>& _sampleBox)
{
	// compute feature template
	//cout << "frame:" << _frame.size() << endl; 
	HaarFeature(_objectBox, featureNum);
	vector<double> initial_scale (featureNum,1.0);
	getFeatureValue(_frame, _sampleBox, initial_scale);

}