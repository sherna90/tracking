/**
 * Particle Marginal Metropolis-Hastings
 * @file pmmh.cpp
 * @brief visual tracking parameter estimation
 * @author Sergio Hernandez
 */
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "../include/dirichlet.hpp"
#include "../include/gaussian.hpp"
#include "../include/multinomial.hpp"
#include "../include/poisson.hpp"
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
    PMMH(string _firstFrameFilename, string _gtFilename);
    void run(int num_particles);

private:
    double marginal_likelihood(int num_particles,int time_step);
    string FrameFilename, gtFilename, firstFrameFilename;
    void getNextFilename(string& fn);
    Rect updateGroundTruth(Mat frame, string str, bool draw);
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    Rect estimate;
    MatND reference_hist,reference_hog;
    Rect2d boundingBox;
};

int main(int argc, char* argv[]){
    int num_particles = 300;
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
        pmmh.run(num_particles);
    }
}

PMMH::PMMH(string _firstFrameFilename, string _gtFilename){
    FrameFilename=_firstFrameFilename;
    firstFrameFilename = _firstFrameFilename;
    Mat current_frame = imread(FrameFilename);
    images.push_back(current_frame);
    while(1){
        getNextFilename(FrameFilename);
        current_frame = imread(FrameFilename );
        if(current_frame.empty()){
            break;
        }else{
          images.push_back(current_frame);
        }
    }
    cout << "Number of images: " << int(images.size()) << endl;
    //Stores all ground-truth strings in a vector
    ifstream gt_file(_gtFilename.c_str(), ios::in);
    string line;
    while (getline(gt_file, line)) gt_vect.push_back(line);
    cout << "Stored " << int(gt_vect.size()) << " ground-truth data" << endl;

    if(images.size() != gt_vect.size()){
        cerr << "There is not the same quantity of images and ground-truth data" << endl;
        cerr << "Maybe you typed wrong filenames" << endl;
        exit(EXIT_FAILURE);
    }

}

double PMMH::marginal_likelihood(int num_particles,int time_step){
    particle_filter pmmh_filter(num_particles);
    MatND reference_hist,reference_hog;
    for(int k=0;k <time_step;k++){    
        Mat current_frame = images[k];
        string current_gt = gt_vect[k];
        if(!pmmh_filter.is_initialized()){
            Rect ground_truth=updateGroundTruth(current_frame,current_gt,true);
            Mat current_roi = Mat(current_frame,ground_truth);
            calc_hist_hsv(current_roi,reference_hist);
            calc_hog(current_roi,reference_hog);
            pmmh_filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);
        }
        else if(pmmh_filter.is_initialized()){
            pmmh_filter.predict();
            pmmh_filter.update_discrete(current_frame,POISSON_LIKELIHOOD,false);
        }
    }
    return pmmh_filter.marginal_likelihood;
}

void PMMH::run(int num_particles){
    string current_filename;
    MatND reference_hist,reference_hog;
    particle_filter filter(num_particles);
    namedWindow("Tracker");
    for(int t=0;t < (int) images.size();t++){
        cout << "---------------" << endl;
        cout << "Time Step t=" << t << endl;
        current_filename = firstFrameFilename;
        Mat current_frame = images[t];
        string current_gt = gt_vect[t];
        Rect ground_truth=updateGroundTruth(current_frame,current_gt,true);
        if(!filter.is_initialized()){
            Mat current_roi = Mat(current_frame,ground_truth);
            calc_hist_hsv(current_roi,reference_hist);
            calc_hog(current_roi,reference_hog);
            filter.initialize(ground_truth,current_frame.size(),reference_hist,reference_hog);
        }
        else{
            filter.predict();
            //filter.update(current_frame,true);
            filter.update_discrete(current_frame,POISSON_LIKELIHOOD,false);
            filter.estimate(current_frame,false);
            cout << "PMMH Marginal Likelihood : " << marginal_likelihood(num_particles,t) << endl;
        }
        filter.draw_particles(current_frame);
        cout << "Filter Marginal Likelihood : " << filter.marginal_likelihood << endl;
        imshow("Tracker",current_frame);
        waitKey(30); 
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


Rect PMMH::updateGroundTruth(Mat frame, string str, bool draw=false){
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
