#ifndef TEST_PARTICLEFILTER_H
#define TEST_PARTICLEFILTER_H

#include "../include/algorithm_test.hpp"

class TestParticleFilter : public AlgorithmTest{
public:
  TestParticleFilter(ImageGenerator * _img_gen, int _num_particles);
  void run();

  int num_particles;
private:
};

#endif //TEST_PARTICLEFILTER_H
