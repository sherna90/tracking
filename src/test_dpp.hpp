#ifndef TEST_DPP_H
#define TEST_DPP_H

#include "DPP/dpp_tracker.hpp"
#include "utils/utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestDPP{
public:
  TestDPP(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  int num_frames;
  imageGenerator generator;
  double reinit_rate;
  vector<Mat> images;
  vector<string> gt_vec;
};

#endif //TEST_PARTICLEFILTER_H
