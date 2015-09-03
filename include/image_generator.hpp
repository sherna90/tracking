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

class ImageGenerator{
public:
  ImageGenerator();
  ImageGenerator(string _firstFrameFilename, string _groundTruthFile);
  void getNextFilename(string& filename);
  bool isEnded();
  string getFrame();
  string getRegion();
  // report();
  int mode; // 0 for local dataset, 1 for VOT
  int frame_id;
  vector<string> filenames;
  vector<string> ground_truths;
private:
};

#endif // IMAGE_GENERATOR_H
