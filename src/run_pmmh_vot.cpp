/**
 * @file app.cpp
 * @brief visual tracking main application
 * @author Sergio Hernandez
 */

#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "../include/utils.hpp"
#include "../include/pmmh.hpp"
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
#include "../src/vot.h"

class App
{
public:
    VOT vot;
    App();
    void help();
    void run(int num_particles,int fixed_lag,int num_mcmc);
    ~App();

private:
    string FrameFilename,gtFilename;
    Rect updateGroundTruth(Mat frame,string str,bool draw);
    void getNextFilename(string& fn);
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
};

App::App(){

}

App::~App(){

}

int main(int argc, char* argv[]){
    int num_particles=300;
    App app;
    app.run(num_particles,3,3);
    return 0;
}

void App::run(int num_particles,int fixed_lag,int num_mcmc){
    pmmh filter(num_particles,fixed_lag,num_mcmc);
    Rect initialization;
    initialization << vot.region();
    Mat current_frame = imread(vot.frame());
    filter.initialize(current_frame, initialization);

    namedWindow("Tracker");
    while(!vot.end()){
      string image_path = vot.frame();
      if (image_path.empty()) break;
      current_frame = imread(image_path);
      filter.update(current_frame);
      Rect estimate = filter.estimate(current_frame,true);
      cout << estimate << endl;
      vot.report(estimate);
      imshow("Tracker",current_frame);
      waitKey(25);
    }
}

void App::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    //size_t index1 = fn.find_last_of("0");
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index+1, index2-index-1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    string zeros ("000000000");
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix + zeros.substr(0,zeros.length()-1-nextFrameNumberString.length())+nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}


Rect App::updateGroundTruth(Mat frame,string str,bool draw=false){
    const int NUMBER=4;
    Point pt[1][NUMBER];
    size_t index1=0;
    size_t index2=-1;
    Mat imageROI;
    for (int i = 0; i < NUMBER; i++){
        index1=str.find(",",index2+1);
        string str_x1 = str.substr(index2+1, index1-index2-1);
        istringstream iss(str_x1);
        int x1 = 0;
        iss >> x1;
        index2=str.find(",",index1+1);
        string str_y1 = str.substr(index1+1, index2-index1-1);
        istringstream iss2(str_y1);
        int y1 = 0;
        iss2 >> y1;
        pt[0][i].x = cvRound(x1);
        pt[0][i].y = cvRound(y1);
    }

    //Make ground truth rect positive independently of point ordering
    int minx = pt[0][0].x;
    int maxx = pt[0][0].x;
    int miny = pt[0][0].y;
    int maxy = pt[0][0].y;
    for(int i = 0; i < NUMBER; i++){
      if(pt[0][i].x < minx)
        minx = pt[0][i].x;
      if(pt[0][i].x > maxx)
        maxx = pt[0][i].x;
      if(pt[0][i].y < miny)
        miny = pt[0][i].y;
      if(pt[0][i].y > maxy)
        maxy = pt[0][i].y;
    }
    if(draw){
        rectangle( frame, Point(minx, miny), Point(maxx, maxy), Scalar(0,255,0), 1, LINE_AA );
    }
    return Rect(minx,miny,cvRound(maxx-minx),cvRound(maxy-miny));
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
