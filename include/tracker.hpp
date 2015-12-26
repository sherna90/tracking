#ifndef TRACKER_H
#define TRACKER_H

#include "image_generator.hpp"

//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <time.h>

using namespace cv;
using namespace std;

class Tracker
{
public:
    Tracker(string _firstFrameFilename,string _gtFilename);
    Rect stringToRect(string str);
    void run();
private:
    ImageGenerator imageGenerator;
};


#endif //TRACKER_H