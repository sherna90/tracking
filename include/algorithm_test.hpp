#ifndef ALGORITHM_TEST_H
#define ALGORITHM_TEST_H

#include <iostream>
#include "image_generator.hpp"
#include "utils.hpp"

using namespace std;

class AlgorithmTest{
public:
  AlgorithmTest();
  virtual void run()=0;
  Rect stringToRect(string str);

  ImageGenerator * imageGenerator;
  Mat current_frame;

  Performance performance;
  time_t start, end;
  double reinit_rate;
  Rect ground_truth;
  int num_frames;
private:
};

#endif // ALGORITHM_TEST_H
