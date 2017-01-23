#ifndef TEST_DPP_H
#define TEST_DPP_H


class TestDPP{
public:
  TestDPP(string _firstFrameFilename, string _gtFilename);
  void run();
private:
  ImageGenerator image_generator;
  double reinit_rate,num_frames;
};

#endif //TEST_PARTICLEFILTER_H
