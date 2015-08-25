/**
 * Particle Marginal Metropolis-Hastings
 * @file pmmh.cpp
 * @brief visual tracking parameter estimation
 * @author Sergio Hernandez
 */
#include <iostream>
#include <chrono>
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
#include <random>

using namespace cv;
using namespace std;

class PMMH
{
public:
    PMMH(string _firstFrameFilename, string _gtFilename);
    void run(int num_particles,int fixed_lag,int mcmc_steps);

private:
    double marginal_likelihood(int num_particles,int time_step,int fixed_lag,VectorXd theta,VectorXd alpha);
    double gamma_prior(VectorXd x,VectorXd a,double b);
    VectorXd discrete_proposal(VectorXd alpha);
    VectorXd continuous_proposal(VectorXd alpha);
    string FrameFilename,gtFilename;
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
        pmmh.run(num_particles,3,3);
    }
}

PMMH::PMMH(string _firstFrameFilename, string _gtFilename){
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
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

double PMMH::marginal_likelihood(int num_particles,int time_step,int fixed_lag,VectorXd theta_x,VectorXd theta_y){
    particle_filter pmmh_filter(num_particles);
    int start_time;
    (fixed_lag==0) || (time_step<fixed_lag) ? start_time=0 : start_time=time_step-fixed_lag;
    for(int k=start_time;k<=time_step;k++){
        Mat current_frame = images[k].clone();
        string current_gt = gt_vect[k];
        if(!pmmh_filter.is_initialized()){
            Rect ground_truth=updateGroundTruth(current_frame,current_gt,false);
            pmmh_filter.initialize(ground_truth,Size(current_frame.cols,current_frame.rows),reference_hist,reference_hog);
            pmmh_filter.update_model(theta_x,theta_y);
        }
        else if(pmmh_filter.is_initialized()){
            pmmh_filter.predict();
            pmmh_filter.update_discrete(current_frame,MULTINOMIAL_LIKELIHOOD,false);
            //pmmh_filter.update(current_frame,false);
        }
        //cout << "PMMH time step:" << k << endl;
    }
    return pmmh_filter.getMarginalLikelihood();
}

VectorXd PMMH::discrete_proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    double eps= std::numeric_limits<double>::epsilon();
    //for(int i=0;i<alpha.size();i++){
    //    proposal[i]=abs(alpha[i]+unif_rnd(generator));
    //}
    for(int i=0;i<alpha.size();i++){
        normal_distribution<double> random_walk(alpha[i],0.1);
        gamma_distribution<double> color_prior(alpha[i],0.1);
        double val=color_prior(generator);
        proposal[i] = (val>0.0) ? val : eps;
    }
    proposal.normalize();
    return proposal;
}

VectorXd PMMH::continuous_proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    for(int i=0;i<alpha.size();i++){
        normal_distribution<double> random_walk(alpha[i],0.1);
        double val=random_walk(generator);
        proposal[i] = val;
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

void PMMH::run(int num_particles,int fixed_lag,int mcmc_steps){
    particle_filter filter(num_particles);
    int num_frames=(int)images.size();
    Performance track_algorithm;
    Performance particle_filter_algorithm;
    VectorXd theta_x,theta_x_prop,theta_y,theta_y_prop,alpha;
    alpha.setOnes((int)H_BINS*S_BINS);
    //alpha.normalize();
    dirichlet prior=dirichlet(alpha);
    Gaussian pos_prior=Gaussian(0.0,POS_STD);
    Gaussian vel_prior=Gaussian(0.0,VEL_STD);
    Gaussian scale_prior=Gaussian(0.0,SCALE_STD);
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
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
            theta_y=filter.get_discrete_model();
            theta_x=filter.get_continuous_model();
        }
        else if(filter.is_initialized()){
            filter.predict();
            filter.update_discrete(current_frame,MULTINOMIAL_LIKELIHOOD,false);
            filter.draw_particles(current_frame);
            if(k>fixed_lag){
            double forward_filter = marginal_likelihood(num_particles,k,fixed_lag,theta_x,theta_y);
            for(int n=0;n<mcmc_steps;n++){
                theta_y_prop=discrete_proposal(theta_y);
                theta_x_prop=continuous_proposal(theta_x);
                //cout << theta_x_prop.transpose() << endl;
                double proposal_filter = marginal_likelihood(num_particles,k,fixed_lag,theta_x_prop,theta_y_prop);
                double acceptprob = proposal_filter - forward_filter;
                acceptprob+=prior.log_likelihood(theta_y_prop)-prior.log_likelihood(theta_y);
                acceptprob+=pos_prior.log_likelihood(theta_x_prop(0))-pos_prior.log_likelihood(theta_x(0));
                acceptprob+=vel_prior.log_likelihood(theta_x_prop(1))-vel_prior.log_likelihood(theta_x(1));
                acceptprob+=scale_prior.log_likelihood(theta_x_prop(2))-scale_prior.log_likelihood(theta_x(2));
                double u=unif_rnd(generator);
                if( log(u) < acceptprob){
                    //cout << "u : " << log(u) << ",alpha : " <<  acceptprob << endl;
                    theta_y=theta_y_prop;
                    theta_x=theta_x_prop;
                    filter.update_model(theta_x_prop,theta_y_prop);
                    forward_filter=proposal_filter;
                    }
                }
            }
        }
        Rect estimate=filter.estimate(current_frame,true);
        particle_filter_algorithm.calc(ground_truth,estimate);
        imshow("Tracker",current_frame);
        waitKey(1);
    }
    cout << "PMMH algorithm >> " <<"average precision:" << particle_filter_algorithm.get_avg_precision()/num_frames << ",average recall:" << particle_filter_algorithm.get_avg_recall()/num_frames << endl;
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
