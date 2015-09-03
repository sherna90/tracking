#ifndef ALGORITHM_TEST_H
#define ALGORITHM_TEST_H

#include <iostream>
#include "../include/image_generator.hpp"

using namespace std;

class AlgorithmTest{
public:
  AlgorithmTest();
  virtual void run()=0;
  Rect stringToRect(string str);

  ImageGenerator * imageGenerator;
  Mat current_frame;
private:
};

#endif // ALGORITHM_TEST_H
