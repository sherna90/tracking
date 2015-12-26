#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

class imageGenerator{
public:
  imageGenerator();
  imageGenerator(string _firstFrameFilename, string _groundTruthFile);
  bool hasEnded();
  void moveNext();
  Mat getFrame();
  Rect getRegion();
  int getDatasetSize();
  vector<Mat> images;
  vector<string> ground_truth;
  Rect stringToRect(string str);
private:
  int frame_id;
  void getNextFilename(string& filename);

};

#endif // IMAGE_GENERATOR_H
