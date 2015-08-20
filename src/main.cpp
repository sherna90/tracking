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

class App
{
public:
    App(string _firstFrameFilename,string _gtFilename);
    void help();
    void run(int);

private:
    string FrameFilename,gtFilename;
    Rect updateGroundTruth(Mat frame,string str,bool draw);
    void getNextFilename(string& fn);
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    Ptr<Tracker> tracker; //added

};

int main(int argc, char* argv[]){
    int num_particles=300;
    if(argc != 5) {
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
        if(strcmp(argv[3], "-gt") == 0) {
            _gtFilename=argv[4];
        }
        else{
            cerr <<"No ground truth given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        App app(_firstFrameFilename,_gtFilename);
        app.run(num_particles);
    }
}

App::App(string _firstFrameFilename, string _gtFilename){
    FrameFilename = _firstFrameFilename;
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

void App::run(int num_particles){
    particle_filter filter(num_particles);
    MatND reference_hist,reference_hog;
    Rect2d boundingBox; //added
    int num_frames=(int)images.size();
    string track_algorithm_selected="MIL";
    tracker = Tracker::create( track_algorithm_selected );
    Performance track_algorithm;
    Performance particle_filter_algorithm;
    namedWindow("Tracker");
    for(int k=0;k <num_frames;++k){    
        Mat current_frame = images[k].clone();
        string current_gt = gt_vect[k];
        Rect ground_truth=updateGroundTruth(current_frame,current_gt,true);
        if(!filter.is_initialized()){
            Mat current_roi = Mat(current_frame,ground_truth);
            calc_hist_hsv(current_roi,reference_hist);
            calc_hog(current_roi,reference_hog);         
            filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);
            boundingBox.x = ground_truth.x;
            boundingBox.y = ground_truth.y;
            boundingBox.width = ground_truth.width;
            boundingBox.height = ground_truth.height;
            tracker->init( current_frame, boundingBox );
        }
        else if(filter.is_initialized()){
            filter.predict();
            filter.update_discrete(current_frame,MULTINOMIAL_LIKELIHOOD,true);
	        //filter.update(current_frame,true);
            filter.draw_particles(current_frame);
            tracker->update( current_frame, boundingBox );
        }
        Rect estimate=filter.estimate(current_frame,true);
        // fixed-lag backward pass
        particle_filter_algorithm.calc(ground_truth,estimate);
        Rect IntboundingBox;
        IntboundingBox.x = (int)boundingBox.x;
        IntboundingBox.y = (int)boundingBox.y;
        IntboundingBox.width = (int)boundingBox.width;
        IntboundingBox.height = (int)boundingBox.height;
        track_algorithm.calc(ground_truth,IntboundingBox);
        //cout << "time : " << k << endl;
        //cout << current_frame.size() << endl;
        imshow("Tracker",current_frame);
        waitKey(30); 
    }
    cout << "track algorithm >> " << "average precision:" << track_algorithm.get_avg_precision()/num_frames << ",average recall:" << track_algorithm.get_avg_recall()/num_frames << endl;
    cout << "particle filter algorithm >> " <<"average precision:" << particle_filter_algorithm.get_avg_precision()/num_frames << ",average recall:" << particle_filter_algorithm.get_avg_recall()/num_frames << endl;
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
