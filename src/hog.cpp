/**
 * @file hist.cpp
 * @brief texture histogram handling
 * @author Sergio Hernandez
 */
 #include "include/hog.h"

using namespace cv;
using namespace std;

void calc_hog(Mat& image)
{
    int num_persons=0;
    ocl::setUseOpenCL(true);
    static const Size win_size = Size(48,48*2);
    static const Size training_padding=Size(0, 0);
    static const Size win_stride = Size(8, 8);
    HOGDescriptor descriptor(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
    descriptor.setSVMDetector( HOGDescriptor::getDaimlerPeopleDetector());
    Mat part_hog;
    vector<Rect> locations;
    if(image.cols>0 && image.rows>0){
        resize(image,part_hog,descriptor.winSize,0,0,INTER_LINEAR);
        descriptor.detectMultiScale(part_hog,locations,0, win_stride,training_padding, 1, 0);
        num_persons=(int)locations.size();
    }
    else num_persons=0;
}