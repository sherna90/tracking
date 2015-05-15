/**
 * @file hist.cpp
 * @brief color histogram handling
 * @author Sergio Hernandez
 */
 #include "../include/hist.hpp"

using namespace cv;
using namespace std;

void calc_hist_hsv(Mat& image, Mat& hist)
{
    int hist_size[] = { H_BINS, S_BINS };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    Mat hsv_base;
    cvtColor( image, hsv_base, COLOR_BGR2HSV );
    calcHist(&hsv_base, 1, channels, Mat(), hist, 2, hist_size, ranges, true, false);
    //normalize(hist, hist, 0, 10, NORM_MINMAX, -1, Mat());
}

void calc_hog(Mat& image,Mat& hist)
{
    ocl::setUseOpenCL(true);
    Mat part_hog;
    vector<float> descriptors;
    vector<Point> points;
    HOGDescriptor descriptor;
    descriptor.winSize=Size(48,96);
    descriptor.nbins=32;
    if(image.cols>0 && image.rows>0){
        resize(image,part_hog,descriptor.winSize,0,0,INTER_LINEAR);
        cvtColor(part_hog, part_hog, COLOR_RGB2GRAY);
        descriptor.compute(part_hog,descriptors,Size(0,0), Size(0,0),points);
        hist.create(1,descriptors.size(),CV_32FC1);
        for(unsigned int i=0;i<descriptors.size();i++){
            hist.at<float>(0,i)=descriptors.at(i);
        }
    }
    //normalize(hist, hist, 0, 1, NORM_MINMAX);
}



