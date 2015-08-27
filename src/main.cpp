/**
 * @file app.cpp
 * @brief visual tracking main application
 * @author Sergio Hernandez
 */

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp> //added
#include "../include/particle_filter.hpp"
#include "../include/utils.hpp"

//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

using namespace cv;
using namespace std;

#define VOT_RECTANGLE
#include "vot.h"

class App
{
public:
    VOT vot;
    particle_filter filter(int _n_particles);

    App(int num_particles);
    void help();
    void run();
    ~App();

private:
    int num_particles;
};

int main(int argc, char* argv[]){
    int num_particles=300;
    App app(num_particles);
    app.run();
    return 0;
}

App::App(int _num_particles){
    num_particles = _num_particles;
}

App::~App(){
  //delete &vot;
}

void App::run(){
    Rect initialization;
    initialization << vot.region();
    Mat initial_frame = imread(vot.frame());
    particle_filter filter(num_particles);
    filter.initialize(initial_frame, initialization);
    //namedWindow("Tracker");
    while(!vot.end()){
        string image_path = vot.frame();
        if (image_path.empty()) break;
        Mat current_frame = imread(image_path);
        filter.predict();
        filter.update_discrete(current_frame);
        filter.draw_particles(current_frame);
        Rect estimate = filter.estimate(current_frame,false);
        vot.report(estimate);
        //imshow("Tracker",current_frame);
        //waitKey(25);
    }
}
