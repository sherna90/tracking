/**
 * @file app.cpp
 * @brief visual tracking main application 
 * @author Sergio Hernandez
 */

#include <opencv2/video/tracking.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "include/hist.h"
#include "include/particle_filter.h"
//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

class App
{
public:
    App(string _firstFrameFilename,string _gtFilename);
    void help();
    void run();

private:
    string firstFrameFilename,gtFilename;
    void updateGroundTruth(Mat frame,string str,bool draw);
    void getNextFilename(string& fn);
    Rect intersect(Rect r1, Rect r2);
    Mat current_frame,current_roi; 
    Rect ground_truth,estimate;
    MatND reference_hist,reference_hog;
    int keyboard;
};



int main(int argc, char* argv[]){
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
        app.run();
    }
}

App::App(string _firstFrameFilename, string _gtFilename){
    firstFrameFilename=_firstFrameFilename;
    gtFilename=_gtFilename;  
}

void App::run(){
    current_frame = imread(firstFrameFilename);
    ifstream groundtruth; 
    groundtruth.open(gtFilename);
    string current_filename(firstFrameFilename),current_gt;
    if(current_frame.empty()){
        //error in opening the first image
        cerr << "Unable to open first image frame: " << firstFrameFilename << endl;
        exit(EXIT_FAILURE);
    }
    particle_filter filter(300);
    Rect intersection;
    double avg_precision=0.0,avg_recall=0.0,ratio,num_frames=0.0;
    namedWindow("Tracker");
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        groundtruth >> current_gt;
        num_frames++;
        if(!filter.is_initialized())
        {
            updateGroundTruth(current_frame,current_gt,true);
            current_roi = Mat(current_frame,ground_truth);
            calc_hist_hsv(current_roi,reference_hist);
            calc_hog(current_roi,reference_hog);
            filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows));
        }
        else if(filter.is_initialized())
        {
            updateGroundTruth(current_frame,current_gt,true);
            filter.predict(Size(current_frame.cols,current_frame.rows));
            filter.update(current_frame,reference_hist,reference_hog);
            //filter.update(current_frame,reference_hist);
            filter.draw_particles(current_frame); 
            estimate=filter.estimate(current_frame,true); 
            intersection=ground_truth & estimate;
            int true_positives,false_positives,false_negatives;
            ratio = double(intersection.area())/double(ground_truth.area());
            if(ratio==1.0){ 
                true_positives=ground_truth.area();
                false_negatives=0;
                false_positives=0;
            }
            else if(ratio>1.0){
                true_positives=ground_truth.area();
                false_negatives=0;
                false_positives=estimate.area()-ground_truth.area();   
            }
            else if(ratio<1.0){
                true_positives=intersection.area();
                false_negatives=ground_truth.area()-intersection.area();
                estimate.area()>0?false_positives=estimate.area()-intersection.area():false_positives=1;   
            }
            //cout << "ratio:" << ratio << ",tp:" << true_positives << ",fp:" << false_positives << ",fn:"<<false_negatives<< ",precision:"<<double(true_positives)/double(true_positives+false_positives)<<endl;
            avg_precision+=double(true_positives)/double(true_positives+false_positives); 
            avg_recall+=double(true_positives)/double(true_positives+false_negatives); 
        }     
        imshow("Tracker", current_frame);
        keyboard = waitKey( 30 );
        getNextFilename(current_filename);
        current_frame = imread(current_filename);
        if(current_frame.empty()){
            cout << "average precision:" << avg_precision/num_frames << ",average recall:" << avg_recall/num_frames << endl;         
            exit(EXIT_FAILURE);
        }
    }
     
           
}

void App::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    size_t index1 = fn.find_last_of("0");
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

void App::updateGroundTruth(Mat frame,string str,bool draw=false){
    const int NUMBER=4;
    Point pt[1][NUMBER];
    size_t index1=0;
    size_t index2=-1;
    Mat imageROI;
    for (int i = 0; i < NUMBER; i++)
    {
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
    if(draw) {
        rectangle( frame, pt[0][1], pt[0][3], Scalar(0,255,0), 1, LINE_AA );
    }
    ground_truth=Rect(pt[0][1].x,pt[0][1].y,cvRound(pt[0][3].x-pt[0][1].x),cvRound(pt[0][3].y-pt[0][1].y));
}

Rect App::intersect(Rect r1, Rect r2) 
{ 
/*    Rect intersection; 
    // find overlapping region 
    intersection.x = (r1.x < r2.x) ? r2.x : r1.x; 
    intersection.y = (r1.y < r2.y) ? r2.y : r1.y; 
    intersection.width = (r1.x + r1.width < r2.x + r2.width) ? 
        r1.x + r1.width : r2.x + r2.width; 
    intersection.width -= intersection.x; 
    intersection.height = (r1.y + r1.height < r2.y + r2.height) ? 
        r1.y + r1.height : r2.y + r2.height; 
    intersection.height -= intersection.y; 

    // check for non-overlapping regions 
    if ((intersection.width <= 0) || (intersection.height <= 0)) { 
        intersection = Rect(0, 0, 0, 0); 
    } */
    return r1 | r2; 
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
