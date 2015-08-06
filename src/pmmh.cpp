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

using namespace cv;
using namespace std;



class PMMH
{
public:
    PMMH(string _firstFrameFilename);
    void run(int num_particles);

private:
    string FrameFilename,gtFilename;
    void getNextFilename(string& fn);
    Mat current_frame;
    vector<Mat>  images;
};



int main(int argc, char* argv[]){
    if(argc != 3) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string _firstFrameFilename,_gtFilename;
        if(strcmp(argv[1], "-img") == 0) {
            _firstFrameFilename=argv[2];
        }
        else{
            cerr <<"No images given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        PMMH pmmh(_firstFrameFilename);
    }
}

PMMH::PMMH(string _firstFrameFilename){
    FrameFilename=_firstFrameFilename;
    current_frame = imread(FrameFilename);
    images.push_back(current_frame);
    while(1){
        getNextFilename(FrameFilename);
        current_frame = imread(FrameFilename);
        images.push_back(current_frame);
        if(current_frame.empty()){
            exit(EXIT_FAILURE);
        }
        cout << images.size() << endl;
    }
}

PMMH::run(int num_particles){
    for(int t=0;t<images.size();t++){
        particle_filter filter(num_particles);
        for(int k=0;k<t;k++){
            current_frame=images[k];
            if(!filter.is_initialized()){
                updateGroundTruth(current_frame,current_gt,true);
                current_roi = Mat(current_frame,ground_truth);
                calc_hist_hsv(current_roi,reference_hist);
                calc_hog(current_roi,reference_hog);
                filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);                
            }
            else if(filter.is_initialized()){
                filter.predict();
                filter.update_discrete(current_frame,MULTINOMIAL_LIKELIHOOD,WITHOUT_HOG);
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






