/**
 * @file app.cpp
 * @brief visual tracking main application 
 * @author Sergio Hernandez
 */

#include <opencv2/video/tracking.hpp>
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

using namespace cv;
using namespace std;

class Performance
{
    private:
        Rect intersection;
        int true_positives, false_positives, false_negatives;
        double avg_precision,avg_recall,ratio;
    public:
        Performance(void);
        void calc(Rect ground_truth, Rect estimate);
        double get_avg_precision(void);
        double get_avg_recall(void);
};


class App
{
public:
    App(string _firstFrameFilename,string _gtFilename);
    void help();
    void run(int,int);

private:
    string firstFrameFilename,gtFilename;
    void updateGroundTruth(Mat frame,string str,bool draw);
    void getNextFilename(string& fn);
    void getPreviousFilename(string& fn);
    Rect intersect(Rect r1, Rect r2);
    Mat current_frame,current_roi; 
    Rect ground_truth,estimate,smoothed_estimate;
    MatND reference_hist,reference_hog;
    int keyboard;
    Rect2d boundingBox; //added
    Ptr<Tracker> tracker; //added

};



int main(int argc, char* argv[]){
    int num_particles=1000,fixed_lag=3;
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
        app.run(num_particles,fixed_lag);
    }
}

App::App(string _firstFrameFilename, string _gtFilename){
    firstFrameFilename=_firstFrameFilename;
    gtFilename=_gtFilename;  
}

void App::run(int num_particles, int fixed_lag){
    current_frame = imread(firstFrameFilename);
    ifstream groundtruth; 
    groundtruth.open(gtFilename.c_str(),ifstream::in);
    string current_filename(firstFrameFilename),current_gt;
    //added to tracking algorithm
    string track_algorithm_selected="MIL";
    tracker = Tracker::create( track_algorithm_selected );
    if(tracker == NULL){
        cout << "Unable to load track algorithm" << endl;
        exit(EXIT_FAILURE);
    }

    if(current_frame.empty()){
        //error in opening the first image
        cerr << "Unable to open first image frame: " << firstFrameFilename << endl;
        exit(EXIT_FAILURE);
    }
    particle_filter filter(num_particles);
    Rect intersection;
    double avg_precision=0.0,avg_recall=0.0,ratio,num_frames=0.0;
    particle_filter filter(300);
    double num_frames=0.0;
    //test object performance
    Performance track_algorithm;
    Performance particle_filter_algorithm;
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
            //added to tracking algorithm
            boundingBox.x = ground_truth.x;
            boundingBox.y = ground_truth.y;
            boundingBox.width = ground_truth.width;
            boundingBox.height = ground_truth.height;
            tracker->init( current_frame, boundingBox );
        }
        else if(filter.is_initialized())
        {
            //add to tracking algorithm
            tracker->update( current_frame, boundingBox );

            updateGroundTruth(current_frame,current_gt,true);
            filter.predict(Size(current_frame.cols,current_frame.rows));
            //filter.update(current_frame,reference_hist,reference_hog);
            //filter.update_dirichlet(current_frame,reference_hist);
            //filter.update_dirichlet(current_frame,reference_hist,reference_hog);
            filter.update(current_frame,reference_hist);
            //filter.draw_particles(current_frame); 
            estimate=filter.estimate(current_frame,true);
            // fixed-lag backward pass

            if(fixed_lag<num_frames){
                string previous_filename=current_filename;
                for(int l=(num_frames);l>(num_frames-fixed_lag);l--){
                    getPreviousFilename(previous_filename);
                }
                Mat previous_frame = imread(previous_filename);
                filter.smoother(fixed_lag);
                smoothed_estimate=filter.smoothed_estimate(current_frame,fixed_lag,true);
                Mat smoothed_roi = Mat(previous_frame,smoothed_estimate);
                calc_hist_hsv(smoothed_roi,smoothed_hist);
            }
            //cout << "-------------------"  << endl; 
            intersection=ground_truth & estimate;
            int true_positives=0,false_positives=0,false_negatives=0;
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
            // cout << "particle weights: time stamp " << num_frames-1;
            // for (int i=0; i<filter.weights[(int)num_frames-1].size();i++){
            //     cout << ", "<< filter.weights[(int)num_frames-1][i] << ",";
            // }
            //cout << "ratio:" << ratio << ",tp:" << true_positives << ",fp:" << false_positives << ",fn:"<<false_negatives<< ",precision:"<<double(true_positives)/double(true_positives+false_positives)<<endl;
            avg_precision+=double(true_positives)/double(true_positives+false_positives); 
            avg_recall+=double(true_positives)/double(true_positives+false_negatives); 
            filter.update_dirichlet(current_frame,reference_hist);
            //filter.update(current_frame,reference_hist);
            filter.draw_particles(current_frame);
            //draw tracker box
            rectangle( current_frame, boundingBox, Scalar( 255, 0, 0 ), 2, 1 ); 
            estimate=filter.estimate(current_frame,true); 
            
            //draw tracker box
            rectangle( current_frame, boundingBox, Scalar( 255, 0, 0 ), 2, 1 ); 
            
            //test performance object
            particle_filter_algorithm.calc(ground_truth,estimate);
            Rect IntboundingBox;
            IntboundingBox.x = (int)boundingBox.x;
            IntboundingBox.y = (int)boundingBox.y;
            IntboundingBox.width = (int)boundingBox.width;
            IntboundingBox.height = (int)boundingBox.height;
            track_algorithm.calc(ground_truth,IntboundingBox);
        }     
        imshow("Tracker", current_frame);
        keyboard = waitKey( 30 );
        getNextFilename(current_filename);
        current_frame = imread(current_filename);
        if(current_frame.empty()){
            //cout << "average precision:" << avg_precision/num_frames << ",average recall:" << avg_recall/num_frames << endl;         
            //test performance object
            cout << "track algorithm >> " << "average precision:" << track_algorithm.get_avg_precision()/num_frames << ",average recall:" << track_algorithm.get_avg_recall()/num_frames << endl;
            cout << "particle filter algorithm >> " <<"average precision:" << particle_filter_algorithm.get_avg_precision()/num_frames << ",average recall:" << particle_filter_algorithm.get_avg_recall()/num_frames << endl;
            exit(EXIT_FAILURE);
        }
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

void App::getPreviousFilename(string& fn){
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

Rect App::intersect(Rect r1, Rect r2) //unused function.
{ 
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

Performance::Performance(void){
    avg_precision=0.0;avg_recall=0.0;
}
void Performance::calc(Rect ground_truth, Rect estimate){
    intersection=ground_truth & estimate;
    true_positives=0;false_positives=0;false_negatives=0;
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
    avg_precision+=double(true_positives)/double(true_positives+false_positives); 
    avg_recall+=double(true_positives)/double(true_positives+false_negatives);
}
double Performance::get_avg_precision(void){
    return avg_precision;
}
double Performance::get_avg_recall(void){
    return avg_recall;
}
