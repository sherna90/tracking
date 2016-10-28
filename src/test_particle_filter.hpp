#ifndef TEST_PARTICLEFILTER_H
#define TEST_PARTICLEFILTER_H


class TestParticleFilter{
public:
  TestParticleFilter(string _firstFrameFilename, string _gtFilename, int _num_particles);
  void run();
private:
  int num_particles;
  ImageGenerator image_generator;
  double reinit_rate,num_frames;
};

#endif //TEST_PARTICLEFILTER_H
