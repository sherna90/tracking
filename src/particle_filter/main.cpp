#include "../../include/particle_filter.hpp"

#define VOT_RECTANGLE
#include "../../include/vot.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    if(argc != 3) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        int _num_particles;

        if(strcmp(argv[1], "-npart") == 0) {
            _num_particles=atoi(argv[2]);
        }
        else{
            _num_particles=300;
        }

        particle_filter tracker(_num_particles);

        VOT vot;

        Rect initialization;
        initialization << vot.region();
        Mat image = imread(vot.frame());
        tracker.initialize(image, initialization);

        while (!vot.end()) {
          string imagepath = vot.frame();
          if (imagepath.empty()) break;
          Mat image = imread(imagepath);

          tracker.predict();
          tracker.update(image);
          Rect estimate = tracker.estimate(image, true);

          vot.report(estimate);
        }

    }
}
