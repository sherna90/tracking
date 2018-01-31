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
  imageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile);
  bool hasEnded();
  void moveNext();
  Mat getFrame();
  Rect getRegion();
  int getDatasetSize();
  vector<Mat> images;
  vector<string> ground_truth;
  vector< vector<Rect> > detections;
  Rect stringToRect(string str);
private:
  void readDetections(string str);
  void readGroundTruth(string str);
  int frame_id;
  void getNextFilename(string& filename);

};

#endif // IMAGE_GENERATOR_H
