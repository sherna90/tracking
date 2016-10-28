#ifndef TEST_PMMH_H
#define TEST_PMMH_H

#include "utils/algorithm_test.hpp"

class TestPMMH : public AlgorithmTest{
public:
  TestPMMH(ImageGenerator * _img_gen, int _num_particles, int _fixed_lag, int _num_mcmc);
  void run();

  int num_particles;
  int fixed_lag;
  int num_mcmc;
private:
};

#endif //TEST_PMMH_H
