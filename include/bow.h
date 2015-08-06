#ifndef BOW_H
#define BOW_H

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <fstream>

#include <bowimage.h>


using namespace cv;
using namespace cv::xfeatures2d;

class Bow
{
public:
    Bow();
    void readImages(string imagesfile,string clases);
    void readImage(string imagefile,int clase);
    void kmeans(int clusters);
    bool computeHistograms();
    void printHistograms();


private:
    vector<BowImage> images;
    Mat trainDescriptors;
    Mat vocabulary;




};

#endif // BOW_H
