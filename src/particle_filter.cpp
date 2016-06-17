/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "../include/particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=1.0;
const float SCALE_STD=0.01;
const float  DT=1.0;
const float  THRESHOLD=0.5;
const bool  USE_COLOR=true;
const bool  USE_LBP=false;
const bool  USE_HAAR=true;
#endif 

particle_filter::particle_filter() {
}


particle_filter::~particle_filter() {
    states.clear();
    weights.clear();
}

particle_filter::particle_filter(int _n_particles) {
    states.clear();
    weights.clear();
    haar_likelihood.clear();
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD,POS_STD;
    theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD,SCALE_STD;
    theta_x.push_back(theta_x_scale);
    eps= std::numeric_limits<double>::epsilon();
}


bool particle_filter::is_initialized() {
    return initialized;
}

void particle_filter::reinitialize() {
    initialized=false;
}


void particle_filter::initialize(Mat& current_frame, Rect ground_truth) {
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    marginal_likelihood=0.0;
    states.clear();
    weights.clear();
    estimates.clear();
    haar_likelihood.clear();
    estimates.push_back(ground_truth);
    //cout << "INIT!!!!!" << endl;
    //cout << ground_truth << endl;
    im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    reference_roi=Rect(left, top, right - left, bottom - top);
    LBP lbp( 8, LBP_MAPPING_NONE );
    VectorXd theta_y_color(H_BINS*S_BINS);
    theta_y_color.setZero();
    VectorXd theta_y_lbp(256);
    theta_y_lbp.setZero();
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width && 
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        marginal_likelihood=0.0;
        double weight=1.0/n_particles;
        sampleBox.clear();//important
        theta_y.clear();
        for (int i=0;i<n_particles;i++){
            particle state;
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            float _dw=scale_random_width(generator);
            float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(state.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width+_dw),0),im_size.width);
            _height=MIN(MAX(cvRound(state.height+_dh),0),im_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),im_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),im_size.height);
            if( (_x+_width)<im_size.width 
                && _x>0 
                && (_y+_height)<im_size.height 
                && _y>0 
                && _width<im_size.width 
                && _height<im_size.height 
                && _width>0 && _height>0){
                state.x_p=state.x;
                state.y_p=state.y;
                state.width_p=state.width;
                state.height_p=state.height;       
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.scale_p=state.scale;
                state.scale=2*state.scale-state.scale_p+scale_random_width(generator);
            }
            else{
                state.x=reference_roi.x;
                state.y=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
            }
            states.push_back(state);
            weights.push_back(weight);
            ESS=0.0f;   
            Rect box(state.x, state.y, state.width, state.height);
            Mat current_roi = Mat(current_frame,box);
            if(USE_COLOR){
                Mat color_hist;
                calc_hist_hsv(current_roi,color_hist);
                for(int h=0;h<H_BINS;h++)
                    for( int s = 0; s < S_BINS; s++ ){
                    double val=color_hist.at<float>(h, s);
                    theta_y_color[h*S_BINS+s]+=val;
               }
            }
            if(USE_LBP){
                cvtColor(current_roi, current_roi, CV_RGB2GRAY);
                current_roi.convertTo( current_roi, CV_64F );
                lbp.calcLBP( current_roi );
                vector<double> lbp_hist = lbp.calcHist().getHist( false );
                for(unsigned int l=0;l<lbp_hist.size();l++)
                   theta_y_lbp[l]+=lbp_hist.at(l);
            }
            sampleBox.push_back(box);    
        }
        if(USE_HAAR){
            Mat grayImg;
            haar_likelihood.clear();
            cvtColor(current_frame, grayImg, CV_RGB2GRAY);
            haar.init(grayImg,reference_roi,sampleBox);
            VectorXd theta_y_mu(haar.featureNum);
            VectorXd theta_y_sig(haar.featureNum);
            Scalar muTemp;
            Scalar sigmaTemp;
            for (int i=0; i<haar.sampleFeatureValue.rows; i++){
                meanStdDev(haar.sampleFeatureValue.row(i), muTemp, sigmaTemp);
                theta_y_mu[i]=muTemp.val[0];
                theta_y_sig[i]=sigmaTemp.val[0];
                Gaussian haar_feature(theta_y_mu[i],theta_y_sig[i]);  
                haar_likelihood.push_back(haar_feature);
            }
            theta_y.push_back(theta_y_mu);
            theta_y.push_back(theta_y_sig);
        }
        if(USE_COLOR){
            theta_y_color=theta_y_color/sampleBox.size();
            theta_y_color=theta_y_color/theta_y_color.sum();
            //cout << "theta_y_color:" << theta_y_color << endl; 
            theta_y.push_back(theta_y_color);
            color_likelihood = Multinomial(theta_y_color);
        }
        if(USE_LBP){
            theta_y_lbp=theta_y_lbp/sampleBox.size();
            theta_y_lbp=theta_y_lbp/theta_y_lbp.sum();
            //cout << "theta_y_color:" << theta_y_color << endl; 
            theta_y.push_back(theta_y_lbp);
            lbp_likelihood = Multinomial(theta_y_lbp);
        }
        initialized=true;
    }
}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    if(initialized==true){
        sampleBox.clear();//important
        time_stamp++;
        vector<particle> tmp_new_states;
        for (int i=0;i<n_particles;i++){
            particle state=states[i];
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            float _dw=scale_random_width(generator);
            float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(state.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width+_dw),0),im_size.width);
            _height=MIN(MAX(cvRound(state.height+_dh),0),im_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),im_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),im_size.height);
            if( (_x+_width)<im_size.width 
                && _x>0 
                && (_y+_height)<im_size.height 
                && _y>0 
                && _width<im_size.width 
                && _height<im_size.height 
                && _width>0 && _height>0){
                state.x_p=state.x;
                state.y_p=state.y;
                state.width_p=state.width;
                state.height_p=state.height;       
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.scale_p=state.scale;
                state.scale=2*state.scale-state.scale_p+scale_random_width(generator);
            }
            else{
                state.x=reference_roi.x;
                state.y=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
            }
            Rect box(state.x, state.y, state.width, state.height);
            sampleBox.push_back(box);
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
            tmp_new_states.push_back(state);
        }
        states.swap(tmp_new_states);
        tmp_new_states = vector<particle>();
    }
}


void particle_filter::draw_particles(Mat& image, Scalar color=Scalar(0,255,255)){
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        Point pt1,pt2;
        pt1.x=cvRound(state.x);
        pt1.y=cvRound(state.y);
        pt2.x=cvRound(state.x+state.width);
        pt2.y=cvRound(state.y+state.height);
        rectangle( image, pt1,pt2, color, 1, LINE_AA );
    }
}

Rect particle_filter::estimate(Mat& image,bool draw=false){
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0;
    Rect estimate;
    for (int i=0;i<n_particles;i++){
        particle state=states.at(i);
        double weight=weights.at(i);
        _x+=(weight*state.x > 0 && weight*state.x < im_size.width) ? weight*state.x : float(1.0/n_particles)*state.x; 
        _y+=(weight*state.y > 0 && weight*state.y < im_size.height) ? weight*state.y : float(1.0/n_particles)*state.y; 
        _width+=(weight*state.width > 0 && weight*state.width < im_size.width) ? weight*state.width : float(1.0/n_particles)*reference_roi.width; 
        _height+=(weight*state.height > 0 && weight*state.height < im_size.height) ? weight*state.height : float(1.0/n_particles)*reference_roi.height; 
        //cout << "weight:" << weight << endl;
        //cout << "ref x:" << reference_roi.x << ",y:" << reference_roi.y <<",w:" << reference_roi.width <<",h:" << reference_roi.height << endl;
        //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
    }
    Point pt1,pt2;
    pt1.x=cvRound(_x);
    pt1.y=cvRound(_y);
    _width=cvRound(_width);
    _height=cvRound(_height);
    pt2.x=cvRound(pt1.x+_width);
    pt2.y=cvRound(pt1.y+_height); 
    if(pt2.x<im_size.width && pt1.x>=0 && pt2.y<im_size.height && pt1.y>=0){
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
        estimate=Rect(pt1.x,pt1.y,_width,_height);
    }
    estimates.push_back(estimate);
    return estimate;

}


void particle_filter::update(Mat& image)
{
    vector<double> tmp_weights;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        if (state.width < 0 || state.width>image.cols){
          state.width = reference_roi.width;
        }
        if (state.height < 0 || state.height>image.rows){
          state.height = reference_roi.width;
        }
        if (state.x < 0 || state.x>image.cols){
          state.x = reference_roi.x;
        }
        if (state.y < 0 || state.y>image.rows){
          state.y = reference_roi.y;
        }
        double weight=weights[i];
        Rect box=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        Mat current_roi = Mat(image,box);
        if(USE_HAAR){
        double prob_haar=0.0f;
            Mat grayImg;
            cvtColor(image, grayImg, CV_RGB2GRAY);
            haar.getFeatureValue(grayImg,sampleBox);
            for(int j=0;j<haar.featureNum;j++){
                //cout << haar.featureNum << "," << i << "," << j << endl; 
                float haar_prob=haar.sampleFeatureValue.at<float>(j,i);
                prob_haar += haar_likelihood.at(j).log_likelihood(haar_prob);
            }
            weight+=prob_haar-log(haar.featureNum);
        }
        if(USE_COLOR){
            Mat color_hist;
            calc_hist_hsv(current_roi,color_hist);
            VectorXd theta_y_color(H_BINS*S_BINS);
            theta_y_color.setZero();
            for(int h=0;h<H_BINS;h++)
                for( int s = 0; s < S_BINS; s++ ){
                    double val=color_hist.at<float>(h, s);
                    theta_y_color[h*S_BINS+s]=val;
               }
            weight+=color_likelihood.log_likelihood(theta_y_color);
        }
        if(USE_LBP){
            LBP lbp( 8, LBP_MAPPING_NONE );
            cvtColor(current_roi, current_roi, CV_RGB2GRAY);
            current_roi.convertTo( current_roi, CV_64F );
            lbp.calcLBP( current_roi );
            VectorXd theta_y_lbp(256);
            theta_y_lbp.setZero();          
            vector<double> lbp_hist = lbp.calcHist().getHist( false );
            for(unsigned int l=0;l<lbp_hist.size();l++)
                theta_y_lbp[l]+=lbp_hist.at(l);
            weight+=lbp_likelihood.log_likelihood(theta_y_lbp);
        }      
        tmp_weights.push_back(weight);
    }
    weights.swap(tmp_weights);
    tmp_weights.clear();
    resample();
}


void particle_filter::resample(){
    vector<double> cumulative_sum(n_particles);
    vector<double> normalized_weights(n_particles);
    vector<double> new_weights(n_particles);
    vector<double> squared_normalized_weights(n_particles);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    double logsumexp=0.0;
    double max_value = *max_element(weights.begin(), weights.end());
    for (unsigned int i=0; i<weights.size(); i++) {
        new_weights[i]=exp(weights[i]-max_value);
        logsumexp+=new_weights[i];
    }
    double norm_const=max_value+log(logsumexp);
    for (unsigned int i=0; i<weights.size(); i++) {
        normalized_weights.at(i) = exp(weights.at(i)-norm_const);
    }
    for (unsigned int i=0; i<weights.size(); i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    marginal_likelihood+=norm_const-log(n_particles); 
    ESS=(1.0f/sum_squared_weights[0])/n_particles;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> new_states;
        for (int i=0; i<n_particles; i++) {
            double uni_rand = unif_rnd(generator);
            vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle state=states[ipos];
            new_states.push_back(state);
            weights.at(i)=log(1.0f/n_particles);
        }
        states.swap(new_states);
    }
    else{
        weights.swap(new_weights);
    }
    cumulative_sum.clear();
    normalized_weights.clear();
    new_weights.clear();
    squared_normalized_weights.clear();
}

float particle_filter::getESS(){
    return ESS;
}

void particle_filter::update_model(vector<VectorXd> theta_x_new,vector<VectorXd> theta_y_new){
    theta_x.clear();
    theta_y.clear();
    haar_likelihood.clear();
    VectorXd theta_x_pos=theta_x_new.at(0);
    VectorXd theta_x_scale=theta_x_new.at(1);
    theta_x.push_back(theta_x_pos);
    theta_x.push_back(theta_x_scale);
    VectorXd theta_y_mu=theta_y_new.at(0);
    VectorXd theta_y_sig=theta_y_new.at(1);
    VectorXd theta_y_color=theta_y_new.at(2);
    for (int i=0; i<haar.featureNum; i++){
        Gaussian haar_feature(theta_y_mu[i],theta_y_sig[i]);  
        haar_likelihood.push_back(haar_feature);
    }
    theta_y.push_back(theta_y_mu);
    theta_y.push_back(theta_y_sig);
    theta_y.push_back(theta_y_color);
    color_likelihood = Multinomial(theta_y_color);
}


vector<VectorXd> particle_filter::get_dynamic_model(){
    return theta_x;
}

vector<VectorXd> particle_filter::get_observation_model(){
    return theta_y;
}
double particle_filter::getMarginalLikelihood(){
    return marginal_likelihood;
}
