#ifndef IMAGEBOW_H
#define IMAGEBOW_H


#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>



using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class BowImage
{
public:

    BowImage();
    Mat computeimage();
    Mat getDescriptors() const;
    void setDescriptors(const Mat &value);

    string getFile() const;
    void setFile(const string &value);

    int getClase() const;
    void setClase(int value);

    Mat getImg() const;
    void setImg(const Mat &value);


    Mat getHistogram() const;
    void setHistogram(const Mat &value);

    std::vector<KeyPoint> getKeypoints() const;
    void setKeypoints(const std::vector<KeyPoint> &value);

private:

    Mat descriptors;
    std::vector<KeyPoint> keypoints;
    string file;
    int clase;
    Mat img;
    Mat histogram;
    Ptr<SURF> detector;
    Ptr<SURF> extractor;
};

#endif // IMAGEBOW_H
