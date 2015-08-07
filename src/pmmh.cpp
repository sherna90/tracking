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
    PMMH(string _firstFrameFilename, string _gtFilename);
    void run(int num_particles, int fixed_lag);

private:
    string FrameFilename, gtFilename, firstFrameFilename;
    void getNextFilename(string& fn);
    void getPreviousFilename(string& fn);
    void updateGroundTruth(Mat frame, string str, bool draw);
    Mat current_frame;
    Mat current_roi;
    Mat fgmask; // Background subtraction mask
    Mat segm;
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    string current_gt;
    Rect ground_truth;
    Rect estimate;
    Rect smoothed_estimate;

    MatND reference_hist,reference_hog,smoothed_hist;

    Ptr<BackgroundSubtractor> fgbg;

    Rect2d boundingBox;

    Ptr<Tracker> tracker;

    queue<Rect> ground_truth_stack;
};

int main(int argc, char* argv[]){
    int num_particles = 300;
    int fixed_lag = 3;
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
        pmmh.run(num_particles, fixed_lag);
    }
}

PMMH::PMMH(string _firstFrameFilename, string _gtFilename){
    FrameFilename=_firstFrameFilename;
    firstFrameFilename = _firstFrameFilename;
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
    ifstream gt_file(_gtFilename.c_str(), ios::in);
    string line;
    while (getline(gt_file, line))
      gt_vect.push_back(line);
    cout << "Stored " << gt_vect.size() << " ground-truth data" << endl;

    if(images.size() != gt_vect.size()){
      cerr << "There is not the same quantity of images and ground-truth data" << endl;
      cerr << "Maybe you typed wrong filenames" << endl;
      exit(EXIT_FAILURE);
    }

    fgbg = cv::bgsegm::createBackgroundSubtractorGMG(5, 0.7);
}

void PMMH::run(int num_particles, int fixed_lag){
    string track_algorithm_selected="MIL";
    tracker = Tracker::create( track_algorithm_selected );
    if(tracker == NULL){
        cout << "Unable to load track algorithm" << endl;
        exit(EXIT_FAILURE);
    }

    string current_filename;
    for(int t=0;t < (int) images.size();t++){
        cout << endl << "Evaluating up to time t=" << t << endl;
        current_filename = firstFrameFilename;
        particle_filter filter(num_particles);
        double num_frames=0.0;

        Performance track_algorithm;
        Performance particle_filter_algorithm;
        Performance smoother_algorithm;
        namedWindow("Tracker");
        for(int k=0;k<t;k++){
            current_frame = images[k];
            current_gt = gt_vect[k];

            fgbg->apply(current_frame, fgmask);
            current_frame.convertTo(segm, CV_8U, 0.5);
            add(current_frame, Scalar(100, 100, 0), segm, fgmask);

            num_frames++;
            if(!filter.is_initialized()){
                updateGroundTruth(current_frame,current_gt,true);

                current_roi = Mat(current_frame,ground_truth);
                Mat roi_mask = Mat(fgmask,boundingBox);

                calc_hist_hsv(current_roi,reference_hist);
                calc_hog(current_roi,reference_hog);
                filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);

                //Tracker is initialized
                boundingBox.x = ground_truth.x;
                boundingBox.y = ground_truth.y;
                boundingBox.width = ground_truth.width;
                boundingBox.height = ground_truth.height;
                tracker->init( current_frame, boundingBox );
            }
            else if(filter.is_initialized()){
                tracker->update( current_frame, boundingBox );
                updateGroundTruth(current_frame,current_gt,true);
                filter.predict();
                filter.update_discrete(current_frame,fgmask,MULTINOMIAL_LIKELIHOOD,WITHOUT_HOG);
                filter.draw_particles(segm);
                estimate=filter.estimate(segm,true);

                ground_truth_stack.push(ground_truth);
                if(fixed_lag<(num_frames)){
                  string previous_filename = current_filename;
                  for(int l=(num_frames);l>(num_frames-fixed_lag);--l){
                      getPreviousFilename(previous_filename);
                  }
                  Mat previous_frame = imread(previous_filename);
                  filter.smoother(fixed_lag);
                  smoothed_estimate=filter.smoothed_estimate(fixed_lag);
                  if(smoothed_estimate.area()>0){
                      filter.update_model(previous_frame,fgmask,smoothed_estimate);
                      smoother_algorithm.calc(ground_truth_stack.front(),smoothed_estimate);
                  }
                  ground_truth_stack.pop();
                }
                rectangle( segm, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
                //cout << "GT: " << ground_truth << " E: " << estimate << endl;
                particle_filter_algorithm.calc(ground_truth,estimate);

                Rect IntboundingBox;
                IntboundingBox.x = (int)boundingBox.x;
                IntboundingBox.y = (int)boundingBox.y;
                IntboundingBox.width = (int)boundingBox.width;
                IntboundingBox.height = (int)boundingBox.height;
                track_algorithm.calc(ground_truth,IntboundingBox);
            }
            imshow("Tracker", segm);
            waitKey(30);
            getNextFilename(current_filename);
            current_frame = imread(current_filename);
            if(k == (t-1)){
                //cout << "average precision:" << avg_precision/num_frames << ",average recall:" << avg_recall/num_frames << endl;
                //test performance object
                cout << "Track algorithm >> " << "average precision:" << track_algorithm.get_avg_precision()/num_frames << ", average recall:" << track_algorithm.get_avg_recall()/num_frames << endl;
                cout << "Particle filter algorithm >> " <<"average precision:" << particle_filter_algorithm.get_avg_precision()/num_frames << ", average recall:" << particle_filter_algorithm.get_avg_recall()/num_frames << endl;
                cout << "Smoothing algorithm >> " <<"average precision:" << smoother_algorithm.get_avg_precision()/num_frames << ", average recall:" << smoother_algorithm.get_avg_recall()/num_frames << endl;
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

void PMMH::getPreviousFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index+1, index2-index-1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber - 1);
    string zeros ("000000000");
    string previousFrameNumberString = oss.str();
    string previousFrameFilename = prefix + zeros.substr(0,zeros.length()-1-previousFrameNumberString.length())+previousFrameNumberString + suffix;
    fn.assign(previousFrameFilename);
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
    ground_truth=Rect(minx,miny,cvRound(maxx-minx),cvRound(maxy-miny));
}
