#include "local_binary_pattern.hpp"
#include <sstream>


LocalBinaryPattern::LocalBinaryPattern(){
	initialized = false;
	numBlocks = 2;
	numSupportPoints = 8;
	mapping = "u2";
	rad = 2;
	normalizedHist = false;
}

void LocalBinaryPattern::getFeatureValue(Mat& _image, vector<Rect> _sampleBox, bool _isPositiveBox){
	//int xMin, xMax, yMin, yMax;

	Mat Image;
	_image.copyTo(Image);
	Image.convertTo( Image, CV_64F );
	int image_width = Image.cols , image_height = Image.rows;
	LBP lbp( numSupportPoints, LBP::strToType( mapping ) );
    lbp.calcLBP( Image, rad, true );
    //Size size(64,128);
    //resize(subImage, Image, size);
	//equalizeHist(Image, Image); //Equalize Image
	for (unsigned int k = 0; k < _sampleBox.size(); ++k)
	{
		
        int width = _sampleBox.at(k).width, height = _sampleBox.at(k).height;

        vector<double> hist;

        for (int i = 0; i < numBlocks; ++i)
        {
        	for (int j = 0; j < numBlocks; ++j)
        	{
        		Mat mask = Mat::zeros(image_height, image_width, CV_8UC1);
        		int x = _sampleBox.at(k).x;
				int y = _sampleBox.at(k).y;
        		x += width / numBlocks * i;
				y += height / numBlocks * j;
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
