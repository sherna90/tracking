/**
 * @file particle_smoother.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "../include/particle_smoother.hpp"

const float POS_STD=1.0;
const float SCALE_STD=1.0;
const float  DT=1.0;
const float  SIGMA_COLOR=0.1;
const float  SIGMA_SHAPE=0.1;
const float  THRESHOLD=0.7;
const int  DIRICHLET_LIKELIHOOD=0;
const int MULTINOMIAL_LIKELIHOOD=1;
const int POISSON_LIKELIHOOD=2;
const int LIKELIHOOD=MULTINOMIAL_LIKELIHOOD;
const bool HOG=true;

particle_smoother::particle_smoother(int _n_particles) {
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    //rng(0xFFFFFFFF);
}

void particle_smoother::smoother(int fixed_lag){
    smoothing_weights=weights.front();
    vector<float> log_backward_probability(n_particles,0.0f);
    vector<float> normalized_weights(n_particles,0.0f);
    vector<float> sum_weights(n_particles,0.0f);
    static const float LAMBDA_POS = 0.5f*1.0f/(pow(POS_STD,2.0f));
    //cout << "------------------" << endl;
    if(fixed_lag<time_stamp){
        for(int k=time_stamp;k>(time_stamp-fixed_lag);--k){
            for (int j=0;j<n_particles;j++){
                particle state=states[k][j];
                for (int l=0;l<n_particles;l++){
                    particle past_state=states[k-1][l];
                    float sum=LAMBDA_POS*pow(state.x-past_state.x-past_state.dx,2.0);
                    sum+=LAMBDA_POS*pow(state.y-past_state.y-past_state.dy,2.0);
                    log_backward_probability[l] = log(weights[k-1].at(l))-log(2.0f*M_PI)-2.0f*log(POS_STD)-sum;
                }
                float logsumexp=0.0f;
                float max_value = *max_element(log_backward_probability.begin(), log_backward_probability.end());
                for (unsigned int h=0; h<log_backward_probability.size(); h++) {
                    logsumexp+=exp(log_backward_probability[h]-max_value);
                }
                sum_weights[j]=max_value+log(logsumexp);
            }
            
            for (int i=0;i<n_particles;i++){
                particle past_state=states[k-1][i];
                double backward_probability=0.0f;
                for (int j=0;j<n_particles;j++){
                    particle state=states[k][j];
                    float sum=LAMBDA_POS*pow(state.x-past_state.x-past_state.dx,2.0);
                    sum+=LAMBDA_POS*pow(state.y-past_state.y-past_state.dy,2.0);
                    float log_prob=log(weights[k-1].at(i))-log(2.0f*M_PI)-2.0f*log(POS_STD)-sum;
                    backward_probability+=exp(log_prob-sum_weights[j]);
                }
                smoothing_weights.at(i) = weights[k-1].at(i)*backward_probability;
                //cout << backward_probability << endl;
                //smoothing_weights.at(i) = weights[k-1].at(i);

            }
        }
    }
}



Rect particle_smoother::smoothed_estimate(int fixed_lag){
    //smoothing_weights=weights.front();
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0;
    Rect estimate;
    for (int i=0;i<n_particles;i++){
        particle state=states[time_stamp-fixed_lag][i];
        _x+=smoothing_weights[i]*state.x;
        _y+=smoothing_weights[i]*state.y;
        _width+=smoothing_weights[i]*state.width;
        _height+=smoothing_weights[i]*state.height;
    }
    Point pt1,pt2;
    pt1.x=cvRound(_x);
    pt1.y=cvRound(_y);
    pt2.x=cvRound(_x+_width);
    pt2.y=cvRound(_y+_height);
    if(pt2.x<im_size.width && pt1.x>=0 && pt2.y<im_size.height && pt1.y>=0){
        estimate=Rect(pt1.x,pt1.y,cvRound(pt2.x-pt1.x),cvRound(pt2.y-pt1.y));
    }
    else{
        cout << "oops!" << endl;
    }
    return estimate;
}


void particle_smoother::update(Mat& image,Mat& fgmask,bool hog=false)
{
    vector<float> tmp_weights;
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        Mat roi_mask = Mat(fgmask,boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        double bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        double prob = 0.0f;
        if(bc_color != 1.0f ){
            prob = color_lilekihood.likelihood(bc_color);
        }
        float weight=weights[time_stamp-1][i]*prob;
        if(hog){
            calc_hog(part_roi,part_hog);
            if(part_hog.size()==reference_hog.size()){
                double bc_hog = compareHist(reference_hog, part_hog, HISTCMP_BHATTACHARYYA);
                double prob_hog = hog_likelihood.likelihood(bc_hog);
                weight*=prob_hog;
            }
        }
        tmp_weights.push_back(weight);
    }
    weights.push_back(tmp_weights);
    resample(false);
}

float particle_smoother::getESS(){
    return ESS/n_particles;
}


