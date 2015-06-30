#include "bowimage.h"

BowImage::BowImage()
{
    detector = SURF::create( 500 );// 500 o 300 son buenos
    extractor=SURF::create();
//    img=Mat();
//    keypoints=Mat();
//    descriptors=Mat();
}
Mat BowImage::getHistogram() const
{
    return histogram;
}

void BowImage::setHistogram(const Mat &value)
{
    histogram = value;
}
std::vector<KeyPoint> BowImage::getKeypoints() const
{
    return keypoints;
}

void BowImage::setKeypoints(const std::vector<KeyPoint> &value)
{
    keypoints = value;
}


Mat BowImage::getImg() const
{
    return img;
}

void BowImage::setImg(const Mat &value)
{
    img = value;
}

int BowImage::getClase() const
{
    return clase;
}

void BowImage::setClase(int value)
{
    clase = value;
}

string BowImage::getFile() const
{
    return file;
}

void BowImage::setFile(const string &value)
{
    file = value;
}

Mat BowImage::computeimage()
{

    detector->detect(img,keypoints);
    extractor->compute(img,keypoints,descriptors);
    return descriptors;
}

Mat BowImage::getDescriptors() const
{
    return descriptors;
}

void BowImage::setDescriptors(const Mat &value)
{
    descriptors = value;
}


