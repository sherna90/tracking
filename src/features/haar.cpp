#include "haar.hpp"
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
	
	int numRect;
	Rect rectTemp;
	float weightTemp;
	for (int i=0; i<_numFeature; i++)
	{
		//cout << "Feature :" << i << ";" ; 
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
		//int dim = rng.uniform(0, 2);
		//cout <<dim <<",";
		for (int j=0; j<numRect; j++)
		{
			/*if(dim==0){
				rectTemp.width=cvCeil(rectTemp.width/numRect);
				rectTemp.x=cvFloor(rectTemp.x+j*rectTemp.width);
				rectTemp.y=rectTemp.y;
				rectTemp.height=rectTemp.height;
			}
			else if(dim==1){
				rectTemp.width=rectTemp.width;
				rectTemp.x=rectTemp.x;
				rectTemp.height=cvCeil(rectTemp.height/numRect);
				rectTemp.y=cvFloor(rectTemp.y+j*rectTemp.height);
			}*/
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(4.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(4.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);
			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			//weightTemp = (j % 2 == 0) ? -1.0 : 1.0;
			featuresWeight[i].push_back(weightTemp);
   			//cout << weightTemp <<"," << halfRect  ;
		}
		//cout << endl;
	}
}

void Haar::getFeatureValue(Mat& _frame, vector<Rect>& _sampleBox)
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
			float scale_x=(float)reference_roi.width/_sampleBox[j].width;
			float scale_y=(float)reference_roi.height/_sampleBox[j].height;
			//if(j<=(int)_sampleScale.size())
				//scale=(float)_sampleScale[j];
			for (size_t k=0; k<features[i].size(); k++)
			{
				xMin = MIN(MAX(cvRound(_sampleBox[j].x + scale_x*features[i][k].x),0),imageIntegral.cols);
				xMax = MIN(MAX(cvRound(_sampleBox[j].x + scale_x*(features[i][k].x + features[i][k].width)),0),imageIntegral.cols);
				yMin = MIN(MAX(cvRound(_sampleBox[j].y + scale_y*features[i][k].y),0),imageIntegral.rows);
				yMax = MIN(MAX(cvRound(_sampleBox[j].y + scale_y*(features[i][k].y + features[i][k].height)),0),imageIntegral.rows);
				/*cout << "Feature : " << featuresWeight[i][k] ;
				cout << "," << imageIntegral.at<float>(yMin, xMin);
				cout << "," << imageIntegral.at<float>(yMax, xMax);
				cout << "," << imageIntegral.at<float>(yMin, xMax); 
				cout << "," << imageIntegral.at<float>(yMax, xMin)  << endl;*/
				//cout << imageIntegral.size() << "," << yMin << "," << xMin << "," << yMax << "," << xMax;
				if(xMax < _sampleBox[j].x+_sampleBox[j].width && yMax < _sampleBox[j].y+_sampleBox[j].height && yMin > 0 && xMin > 0){
					tempValue += (featuresWeight[i][k]/_sampleBox[j].area()) *
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
	reference_roi=_objectBox;
	HaarFeature(_objectBox, featureNum);
	getFeatureValue(_frame, _sampleBox);
}
