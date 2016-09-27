/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "../include/particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=5.0;
const float SCALE_STD=1.0;
const float  DT=1.0;
const float  THRESHOLD=0.8;
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
    positive_likelihood.clear();
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
    normal_distribution<double> negative_random_pos(0.0,20.0);
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    marginal_likelihood=0.0;
    vector<Rect > negativeBox;
    states.clear();
    weights.clear();
    estimates.clear();
    positive_likelihood.clear();
    negative_likelihood.clear();
    sampleBox.clear();
    sampleScale.clear();
    theta_y.clear();
    estimates.push_back(ground_truth);
    //cout << "INIT!!!!!" << endl;
    //cout << ground_truth << endl;
    im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    reference_roi=Rect(left, top, right - left, bottom - top);
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width && 
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        marginal_likelihood=0.0;
        float weight=log(1.0/n_particles);
        for (int i=0;i<n_particles;i++){
            particle state;
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            //float _dw=scale_random_width(generator);
            //float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(reference_roi.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(reference_roi.y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(reference_roi.width),10.0),im_size.width);
            _height=MIN(MAX(cvRound(reference_roi.height),10.0),im_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),im_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),im_size.height);
            if( (_x+_width)<im_size.width 
                && _x>0 
                && (_y+_height)<im_size.height 
                && _y>0 
                && _width<im_size.width 
                && _height<im_size.height 
                && _width>0 && _height>0){
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width_p=reference_roi.width;
                state.height_p=reference_roi.height;       
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.scale_p=state.scale;
                state.scale=1.0;
            }
            else{
                state.x=reference_roi.x;
                state.y=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width_p=cvRound(reference_roi.width);
                state.height_p=cvRound(reference_roi.height);
                state.scale=1.0;
            }
            states.push_back(state);
            weights.push_back(weight);
            ESS=0.0f;   
            Rect box(state.x, state.y, state.width, state.height);
            sampleBox.push_back(box);
            sampleScale.push_back(state.scale);    
        }
        for (int i=0;i<n_particles;i++){
            float _x,_y;
            float _dx=negative_random_pos(generator);
            float _dy=negative_random_pos(generator);
            _x=MIN(MAX(cvRound(reference_roi.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(reference_roi.y+_dy),0),im_size.height);
            Rect box;
            box.x=_x;
            box.y=_y;
            box.width=cvRound(reference_roi.width);
            box.height=cvRound(reference_roi.height);
            negativeBox.push_back(box);    
        }
        Mat grayImg;
        cvtColor(current_frame, grayImg, CV_RGB2GRAY);
        equalizeHist( grayImg, grayImg );
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
            positive_likelihood.push_back(haar_feature);
        }
        theta_y.push_back(theta_y_mu);
        theta_y.push_back(theta_y_sig);
        haar.getFeatureValue(grayImg,negativeBox,sampleScale);
        for (int i=0; i<haar.featureNum; i++){
            meanStdDev(haar.sampleFeatureValue.row(i), muTemp, sigmaTemp);
            Gaussian haar_feature((double)muTemp.val[0],(double)sigmaTemp.val[0]); 
            negative_likelihood.push_back(haar_feature);
        }
        initialized=true;
    }
}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    sampleBox.clear();//important
    sampleScale.clear();//important
    //cout << "predicted particles!" <<endl;
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states(n_particles);
        for (int i=0;i<n_particles;i++){
            particle state=states[i];
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            float _dw=scale_random_width(generator);
            float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(state.x),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width),10),im_size.width);
            _height=MIN(MAX(cvRound(state.height),10),im_size.height);
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
                state.x=cvRound(2*_x-state.x_p+_dx);
                state.y=cvRound(2*_y-state.y_p+_dy);
                state.width_p=state.width;
                state.height_p=state.height;       
                state.width=cvRound(2*_width-state.width_p+_dw);
                state.height=cvRound(2*_height-state.height_p+_dw);
                state.scale=((float)state.width/(float)reference_roi.width)/2.0f+((float)state.height/(float)reference_roi.height)/2.0f;
            }
            else{
                state.x=state.x;
                state.y=state.y;
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
                state.width_p=cvRound(reference_roi.width);
                state.height_p=cvRound(reference_roi.height);
                state.scale=1.0f;
            }
            Rect box(state.x, state.y, state.width, state.height);
            sampleBox.push_back(box);
            sampleScale.push_back(state.scale);
            //cout << "reference " << reference_roi << endl;
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height <<",scale:" << state.scale << endl;
            tmp_new_states[i]=state;
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
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0,norm=0.0;
    Rect estimate;
    //cout << "estimated particles!" <<endl;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        //double weight=weights.at(i);
        if(state.x>0 && state.x<im_size.width 
            && state.y>0  && state.y<im_size.height 
            && state.width>0 && state.width<im_size.height 
            && state.height>0 && state.height<im_size.height){
            _x+= state.x; 
            _y+= state.y; 
            _width+= state.width; 
            _height+= state.height;
            norm++;
        }
        //else{
        //    cout << "weight:"  << weights[i] <<", x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
        //} 
        //cout << "weight:" << weight << endl;
        //cout << "ref x:" << reference_roi.x << ",y:" << reference_roi.y <<",w:" << reference_roi.width <<",h:" << reference_roi.height << endl;
        //
    }
    Point pt1,pt2;
    pt1.x=cvRound(_x/norm);
    pt1.y=cvRound(_y/norm);
    _width=cvRound(_width/norm);
    _height=cvRound(_height/norm);
    pt2.x=cvRound(pt1.x+_width);
    pt2.y=cvRound(pt1.y+_height); 
    if(pt2.x<im_size.width && pt1.x>=0 && pt2.y<im_size.height && pt1.y>=0){
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
        estimate=Rect(pt1.x,pt1.y,_width,_height);
    }
    //cout << " estimate x:" << estimate.x << ",y:" << estimate.y <<",w:" << estimate.width <<",h:" << estimate.height << endl;
    estimates.push_back(estimate);
    return estimate;

}


void particle_filter::update(Mat& image)
{
    vector<float> tmp_weights;
    //uniform_int_distribution<int> random_feature(0,haar.featureNum-1);
    Mat grayImg;
    cvtColor(image, grayImg, CV_RGB2GRAY);
    equalizeHist( grayImg, grayImg );
    haar.getFeatureValue(grayImg,sampleBox,sampleScale);
    //cout << "updated particles!" <<endl;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        //cout << i << ", x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
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
        float weight=weights[i];
        float prob_haar=0.0f;
        //int feature_index = random_feature(generator);
        for(int j=0;j<haar.featureNum;j++){
            //cout << haar.featureNum << "," << i << "," << j << endl; 
            float haar_prob=haar.sampleFeatureValue.at<float>(j,i);
            prob_haar += positive_likelihood.at(j).log_likelihood(haar_prob)-negative_likelihood.at(j).log_likelihood(haar_prob);
        }
        weight=prob_haar;
        tmp_weights.push_back(weight);
    }
    weights.swap(tmp_weights);
    tmp_weights.clear();
    resample();
}


void particle_filter::resample(){
    vector<float> cumulative_sum(n_particles);
    vector<float> normalized_weights(n_particles);
    vector<float> new_weights(n_particles);
    vector<float> squared_normalized_weights(n_particles);
    uniform_real_distribution<float> unif_rnd(0.0,1.0); 
    float logsumexp=0.0f;
    float max_value = *max_element(weights.begin(), weights.end());
    for (unsigned int i=0; i<weights.size(); i++) {
        new_weights[i]=exp(weights.at(i)-max_value);
        logsumexp+=new_weights[i];
    }
    float norm_const=max_value+log(logsumexp);
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = exp(weights.at(i)-norm_const);
    }
    for (int i=0; i<n_particles; i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
        //cout << " cumsum: " << normalized_weights.at(i) << "," <<cumulative_sum.at(i) << endl;
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    marginal_likelihood=norm_const-log(n_particles); 
    ESS=(1.0f/sum_squared_weights[0])/n_particles;
    //cout << "resampled particles!" << ESS << endl;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> new_states(n_particles);
        for (int i=0; i<n_particles; i++) {
            float uni_rand = unif_rnd(generator);
            vector<float>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            unsigned int ipos = distance(cumulative_sum.begin(), pos);
            particle state=states[ipos];
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
            new_states[i]=state;
            weights[i]=log(1.0f/n_particles);
        }
        states.swap(new_states);
        new_states = vector<particle>();
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
    positive_likelihood.clear();
    VectorXd theta_x_pos=theta_x_new.at(0);
    VectorXd theta_x_scale=theta_x_new.at(1);
    theta_x.push_back(theta_x_pos);
    theta_x.push_back(theta_x_scale);
    VectorXd theta_y_mu=theta_y_new.at(0);
    VectorXd theta_y_sig=theta_y_new.at(1);
    for (int i=0; i<haar.featureNum; i++){
        Gaussian haar_feature(theta_y_mu[i],theta_y_sig[i]);  
        positive_likelihood.push_back(haar_feature);
    }
    theta_y.push_back(theta_y_mu);
    theta_y.push_back(theta_y_sig);
}


vector<VectorXd> particle_filter::get_dynamic_model(){
    return theta_x;
}

vector<VectorXd> particle_filter::get_observation_model(){
    return theta_y;
}
float particle_filter::getMarginalLikelihood(){
    return marginal_likelihood;
}
