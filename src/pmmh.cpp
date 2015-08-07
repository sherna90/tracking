/**
 * @file app.cpp
 * @brief visual tracking main application
 * @author Sergio Hernandez
 */

#include <opencv2/video/tracking.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp> //added
#include "../include/hist.hpp"
#include "../include/particle_filter.hpp"

//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

#include <typeinfo>

using namespace cv;
using namespace std;

class PMMH
{
public:
    PMMH(string _firstFrameFilename, string _gtFilename);
    void run(int num_particles);

private:
    string FrameFilename,gtFilename;
    void getNextFilename(string& fn);
    void updateGroundTruth(Mat frame, string str, bool draw);
    Mat current_frame;
    Mat current_roi;
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    string current_gt;
    Rect ground_truth;

    MatND reference_hist,reference_hog,smoothed_hist;

};

int main(int argc, char* argv[]){
    if(argc != 5) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename, _gtFilename;
        if(strcmp(argv[1], "-img") == 0) {
            _firstFrameFilename=argv[2];
        }
        else{
            cerr <<"No images given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[3], "-gt") == 0) {
            _gtFilename=argv[4];
        }
        else{
            cerr <<"No ground truth given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        PMMH pmmh(_firstFrameFilename, _gtFilename);
        pmmh.run(300);
    }
}

PMMH::PMMH(string _firstFrameFilename, string _gtFilename){
    FrameFilename=_firstFrameFilename;
    current_frame = imread(FrameFilename);
    images.push_back(current_frame);
    while(1){
        getNextFilename(FrameFilename);
        current_frame = imread(FrameFilename);
        if(current_frame.empty()){
            break;
        }else{
          images.push_back(current_frame);
        }
    }
    cout << "Number of images: " << images.size() << endl;
    //Stores all ground-truth strings in a vector
    const char * f = _gtFilename.c_str();
    ifstream gt_file(f, ios::in);
    string line;
    while (getline(gt_file, line))
      gt_vect.push_back(line);
    cout << "Stored " << gt_vect.size() << " ground-truth data" << endl;
}

void PMMH::run(int num_particles){
    for(int t=0;t < (int) images.size();t++){
        particle_filter filter(num_particles);
        for(int k=0;k<t;k++){
            current_frame = images[k];
            current_gt = gt_vect[k];

            if(!filter.is_initialized()){
                updateGroundTruth(current_frame,current_gt,true);
                if(true){
                  //A temporal fix for compatibility with vot2015 dataset
                  ground_truth.x += ground_truth.width;
                  ground_truth.y += ground_truth.height;
                  ground_truth.width *= -1;
                  ground_truth.height *= -1;
                }
                current_roi = Mat(current_frame,ground_truth);
                calc_hist_hsv(current_roi,reference_hist);
                calc_hog(current_roi,reference_hog);
                filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);
            }
            else if(filter.is_initialized()){
                filter.predict();
                //filter.update_discrete(current_frame,MULTINOMIAL_LIKELIHOOD,WITHOUT_HOG);
            }
        }
    }
}

void PMMH::getNextFilename(string& fn){
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

void PMMH::updateGroundTruth(Mat frame, string str, bool draw=false){
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

    if(draw){
        rectangle( frame, pt[0][1], pt[0][3], Scalar(0,255,0), 1, LINE_AA );
    }
    ground_truth=Rect(pt[0][1].x,pt[0][1].y,cvRound(pt[0][3].x-pt[0][1].x),cvRound(pt[0][3].y-pt[0][1].y));
}
