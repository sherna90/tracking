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
    particle_filter marginal_likelihood(
    int num_particles,int time_step,VectorXd alpha);
    double gamma_prior(VectorXd x,VectorXd a,double b);
    VectorXd proposal(VectorXd alpha);
    string FrameFilename, gtFilename, firstFrameFilename;
    void getNextFilename(string& fn);
    Rect updateGroundTruth(Mat frame, string str, bool draw);
    vector<Mat> images;
    //Stores ground-truth data
    vector<string> gt_vect;
    Rect estimate;
    MatND reference_hist,reference_hog;
    Rect2d boundingBox;
    default_random_engine generator;
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

particle_filter PMMH::marginal_likelihood(int num_particles,int time_step,VectorXd alpha){
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
            pmmh_filter.update_model(alpha);
        }
        else if(pmmh_filter.is_initialized()){
            pmmh_filter.predict();
            pmmh_filter.update_discrete(current_frame,POISSON_LIKELIHOOD,false);
        }
    }
    return pmmh_filter;
}

VectorXd PMMH::proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    uniform_int_distribution<int> unif_rnd(-1.0,1.0);
    for(int i=0;i<alpha.size();i++){
        proposal[i]=alpha[i]+unif_rnd(generator);
    }
    return proposal;
}

double PMMH::gamma_prior(VectorXd x, VectorXd a, double b)
{
    double loglike=0.0; 
    for(int i=0;i<a.size();i++){   
        if (x(i) >= 0 || a(i) >= 0 || b >= 0){
            loglike+=-x(i)*b+(a(i)-1.0)*log(x(i))+a(i)*log(b)-lgamma(a(i));
        }
    }
    return loglike;
}

void PMMH::run(int num_particles){
    string current_filename;
    MatND reference_hist,reference_hog;
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    double eps= std::numeric_limits<double>::epsilon();
    namedWindow("Tracker");
    Rect ground_truth=updateGroundTruth(images[0],gt_vect[0],true);
    Mat current_roi = Mat(images[0],ground_truth);
    calc_hist_hsv(current_roi,reference_hist);
    calc_hog(current_roi,reference_hog);
    VectorXd alpha0,alpha,alpha_prop;
    alpha.setOnes(reference_hist.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ ){
            double val=reference_hist.at<float>(h, s);
            gamma_distribution<double> color_prior(val,1.0);
            alpha[h*S_BINS+s] = (val!=0.0) ? color_prior(generator) : eps;
    }
    alpha0=alpha;
    Performance pmmh_algorithm;      
    for(int t=1;t < (int) images.size();t++){
        cout << "---------------" << endl;
        cout << "Time Step t=" << t << endl;
        particle_filter filter = marginal_likelihood(num_particles,t,alpha);
        cout << "Filter Marginal Likelihood : " << filter.marginal_likelihood  << endl;
        Mat current_frame = images[t];
        filter.draw_particles(current_frame);
        Rect estimate=filter.estimate(current_frame,true);
        Rect ground_truth=updateGroundTruth(current_frame,gt_vect[t],true);
        pmmh_algorithm.calc(ground_truth,estimate);
        for(int n=0;n<3;n++){
            alpha_prop=proposal(alpha);
            particle_filter proposal_filter = marginal_likelihood(num_particles,t,alpha_prop);
            double acceptprob = proposal_filter.marginal_likelihood - filter.marginal_likelihood;
            acceptprob+=gamma_prior(alpha_prop,alpha0,1.0)-gamma_prior(alpha,alpha0,1.0);
            double u=unif_rnd(generator);
            if(u < exp(acceptprob)){
                cout << "Proposal Marginal Likelihood : " << proposal_filter.marginal_likelihood  << endl;
                alpha=alpha_prop;
            }
        }
        imshow("Tracker",current_frame);
        waitKey(30); 
    }
    cout << "PMMH algorithm >> " <<"average precision:" << pmmh_algorithm.get_avg_precision()/images.size() << ",average recall:" << pmmh_algorithm.get_avg_recall()/images.size() << endl;
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
