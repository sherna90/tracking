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
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    Ptr<Tracker> tracker; //added

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
  delete &vot;
}

void App::run(){
    Rect initialization;
    initialization << vot.region();
    Mat initial_frame = imread(vot.frame());
    particle_filter filter(num_particles);
    filter.initialize(initial_frame, initialization);
    namedWindow("Tracker");
    while(!vot.end()){
        string image_path = vot.frame();
        if (image_path.empty()) break;
        Mat current_frame = imread(image_path);
        filter.predict();
        filter.update_discrete(current_frame);
        filter.draw_particles(current_frame);
        Rect estimate = filter.estimate(current_frame,true);
        vot.report(estimate);
        imshow("Tracker",current_frame);
        waitKey(25);
    }
}




void App::help(){
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use color tracking provided by "  << endl
    << " OpenCV. You can process both videos (-vid) and images (-img)."             << endl
                                                                                    << endl
    << "Usage:"                                                                     << endl
    << "./tracker {-vid <video filename>|-img <image filename> -gt <groundtruth filename>}"                         << endl
    << "for example: ./tracker -vid video.avi -gt groundtruth.txt"                                           << endl
    << "or: ./tracker -img /data/images/1.png -gt groundtruth.txt"                                           << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}
