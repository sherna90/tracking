#include "../include/local_binary_pattern.hpp"

#include <sstream>


LocalBinaryPattern::LocalBinaryPattern(){
	initialized = false;
	numBlocks = 2;
	numSupportPoints = 8;
	mapping = "u2";
	rad = 2;
	normalizedHist = true;
}

void LocalBinaryPattern::getFeatureValue(Mat& _image, vector<Rect> _sampleBox, bool _isPositiveBox){
	int xMin, xMax, yMin, yMax;
	for (unsigned int k = 0; k < _sampleBox.size(); ++k)
	{
		Rect box = _sampleBox.at(k);

		Size size(50,50);
		//cout << "x: " << box.x << "  y: " << box.y << "  height: " << box.height << "  width: " << box.width << endl;

		xMin = MIN(MAX(box.x,0),_image.cols);
		xMax = MIN(box.x + box.width, _image.cols);
		yMin = MIN(MAX(box.y,0),_image.rows);
		yMax = MIN(box.y + box.height, _image.rows);
		//cout << "xMin: " << xMin << "  xMax: " << xMax << "  yMin: " << yMin << "  yMax: " << yMax << endl;		

		//Mat subImage = _image(Rect(xMin, yMin, xMax-xMin, yMax-yMin));
		Mat auxSubImage = _image(Rect(xMin, yMin, xMax-xMin, yMax-yMin));
		Mat subImage;
		auxSubImage.copyTo(subImage);
		
		resize(subImage, subImage, size);
		equalizeHist(subImage, subImage); //Equalize Image
        
        subImage.convertTo( subImage, CV_64F );
        int width = subImage.cols, height = subImage.rows;

        LBP lbp( numSupportPoints, LBP::strToType( mapping ) );
        lbp.calcLBP( subImage, rad, true );
        Mat mask(height, width, CV_8UC1);
        
        vector<double> hist;
        for (int i = 0; i < numBlocks; ++i)
        {
        	for (int j = 0; j < numBlocks; ++j)
        	{
        		mask = Mat::zeros(height, width, CV_8UC1);
        		int x = width / numBlocks * i;
				int y = height / numBlocks * j;
				int wH = width / numBlocks - numBlocks;
				int hH = height / numBlocks - numBlocks;
				Mat roi( mask, Range( y, y + hH ), Range( x, x + wH ) );
				roi = Scalar( 255 );

				vector<double> histAux = lbp.calcHist( mask ).getHist();
				hist.insert(hist.end(), histAux.begin(), histAux.end());
        	}
        }
        
        if(_isPositiveBox){
	        for (unsigned int i = 0; i < hist.size(); ++i)
	        {
	        	sampleFeatureValue(k,i) = hist[i];
	        }
	    }else{
	    	for (unsigned int i = 0; i < hist.size(); ++i)
	        {
	        	negativeFeatureValue(k,i) = hist[i];
	        }
	    }
	}
	/* size:
    -hf = 32
    -riu2 = 10
    -ri = 36
    -u2 = 59
    */
}

void LocalBinaryPattern::init(Mat& _image, vector<Rect> _sampleBox){
	sampleFeatureValue = MatrixXd(_sampleBox.size(),numBlocks*numBlocks*59);
    negativeFeatureValue = MatrixXd(_sampleBox.size(),numBlocks*numBlocks*59);
    getFeatureValue(_image, _sampleBox);
    initialized=true;
}