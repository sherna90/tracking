/**
 * @file hist.cpp
 * @brief color histogram handling
 * @author Sergio Hernandez
 */
 #include "hist.hpp"

using namespace cv;
using namespace std;

const int H_BINS=3;
const int S_BINS=3;

void calc_hist_hsv(Mat& image,  Mat& Mask, Mat& hist)
{
    int hist_size[] = { H_BINS, S_BINS };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 255 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    Mat hsv_base;
    cvtColor( image, hsv_base, COLOR_BGR2HSV );
    //erode(hsv_base, hsv_base, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) );
    //dilate( hsv_base, hsv_base, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) ); 
    calcHist(&hsv_base, 1, channels, Mask, hist, 2, hist_size, ranges, true, false);
}

void calc_hist_hsv(Mat& image, Mat& hist)
{
    int hist_size[] = { H_BINS, S_BINS };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 255 };
    const float* ranges[] = { h_ranges, s_ranges};
    int channels[] = { 0, 1};
    Mat hsv_base;
    cvtColor( image, hsv_base, COLOR_BGR2HSV );
    calcHist(&hsv_base, 1, channels, Mat(),hist, 2, hist_size, ranges, true, false);
    //normalize(hist, hist,0.0,255, NORM_MINMAX, -1, Mat());
}

void colorReduce(Mat& image, int div)
{    
    int nl = image.rows;                    // number of lines
    int nc = image.cols * image.channels(); // number of elements per line
    for (int j = 0; j < nl; j++)
    {
        // get the address of row j
        uchar* data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++)
        {
            // process each pixel
            data[i] = data[i] / div * div + div / 2;
        }
    }
}

