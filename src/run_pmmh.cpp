/**
 * @file app.cpp
 * @brief visual tracking main application
 * @author Sergio Hernandez
 */

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp> //added
#include "../include/pmmh.hpp"
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
    pmmh filter(int num_particles,int fixed_lag,int mcmc_steps);

    App(int num_particles,int fixed_lag,int mcmc_steps);
    void help();
    void run();
    ~App();

private:
    int num_particles,fixed_lag,mcmc_steps;
};

int main(int argc, char* argv[]){
    int num_particles=100;
    App app(num_particles,3,10);
    app.run();
    return 0;
}

App::App(int _num_particles,int _fixed_lag,int _mcmc_steps){
    num_particles = _num_particles;
    fixed_lag=_fixed_lag;
    mcmc_steps=_mcmc_steps;
}

App::~App(){
  //delete &vot;
}

void App::run(){
    Rect initialization;
    initialization << vot.region();
    Mat initial_frame = imread(vot.frame());
    pmmh filter(num_particles,fixed_lag,mcmc_steps);
    filter.initialize(initial_frame, initialization);
    //namedWindow("Tracker");
    while(!vot.end()){
        string image_path = vot.frame();
        if (image_path.empty()) break;
        Mat current_frame = imread(image_path);
        filter.update(current_frame);
        Rect estimate = filter.estimate(current_frame,false);
        vot.report(estimate);
        //imshow("Tracker",current_frame);
        //waitKey(1);
    }
}
