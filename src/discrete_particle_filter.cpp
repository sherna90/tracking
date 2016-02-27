/**
 * @file discrete_particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "../include/discrete_particle_filter.hpp"



discrete_particle_filter::discrete_particle_filter() {
}


discrete_particle_filter::~discrete_particle_filter() {
    states=vector<particle>();
    weights=vector<double>();
}

discrete_particle_filter::discrete_particle_filter(int _n_particles) {
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    theta_x.resize(2);
    theta_x << POS_STD,SCALE_STD;
    eps= std::numeric_limits<double>::epsilon();
}


bool discrete_particle_filter::is_initialized() {
    return initialized;
}

void discrete_particle_filter::reinitialize() {
    initialized=false;
}


void discrete_particle_filter::initialize(Mat& current_frame, Rect ground_truth) {
    normal_distribution<double> position_random_walk(0.0,theta_x(0));
    normal_distribution<double> scale_random_walk(0.0,theta_x(1));
    marginal_likelihood=0.0;
    states = vector<particle>();
    weights = vector<double>();
    //cout << "INIT!!!!!" << endl;
    //cout << ground_truth << endl;
    im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    reference_roi=Rect(left, top, right - left, bottom - top);
    sampleBox.clear();//important
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width && 
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        Mat current_roi = current_frame(reference_roi).clone();
        calc_hist_hsv(current_roi,reference_hist);
        marginal_likelihood=0.0;
        double weight=1.0/n_particles;
        for (int i=0;i<n_particles;i++){
            particle state;
            state.x=cvRound(reference_roi.x+position_random_walk(generator));
            state.y=cvRound(reference_roi.y+position_random_walk(generator));
            state.x_p=cvRound(reference_roi.x+position_random_walk(generator));;
            state.y_p=cvRound(reference_roi.y+position_random_walk(generator));
            state.width=cvRound(right-left);
            state.height=cvRound(bottom-top);
            state.width_p=cvRound(right-left+scale_random_walk(generator));
            state.height_p=cvRound(bottom-top+scale_random_walk(generator));
            state.scale=1.0f+scale_random_walk(generator);
            state.scale_p=0.0;
            Rect box(state.x, state.y, state.width, state.height);
            sampleBox.push_back(box);
            states.push_back(state);
            weights.push_back(weight);
            ESS=0.0f;       
        }
        theta_y.resize(reference_hist.total());
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ ){
                double val=reference_hist.at<float>(h, s);
                gamma_distribution<double> color_prior(val,1.0);
                //theta_y[h*S_BINS+s] = (val!=0.0) ? color_prior(generator) : eps;
                theta_y[h*S_BINS+s] = (val!=0.0) ? val : eps;
            }    
        theta_y.normalize();
        color_likekihood = Multinomial(theta_y);
        if(HOG){
             Mat grayImg;
             cvtColor(current_frame, grayImg, CV_RGB2GRAY);
             haar.init(grayImg,reference_roi,sampleBox);
             Scalar muTemp;
             Scalar sigmaTemp;
             theta_hog.resize(haar.sampleFeatureValue.cols);
             for (int i=0; i<haar.sampleFeatureValue.rows; i++){
                 meanStdDev(haar.sampleFeatureValue.row(i), muTemp, sigmaTemp);
                 theta_hog[i]=muTemp.val[0];
             }
             theta_hog.normalize();
             hog_likelihood=Multinomial(theta_hog);
/*            calc_hog(current_roi,reference_hog);
            theta_hog.resize(reference_hog.total());
            for(unsigned int j = 0; j < reference_hog.total(); j++ ){
                double val=reference_hog.at<float>(0,j);
                theta_hog[j] = (val!=0.0) ? val : eps;
            }
            theta_hog.normalize();
            hog_likelihood=Multinomial(theta_hog);   */      
        }
        initialized=true;
    }
}



void discrete_particle_filter::predict(){
    normal_distribution<double> position_random_walk(0.0,theta_x(0));
    normal_distribution<double> scale_random_walk(0.0,theta_x(1));
    if(initialized==true){
        sampleBox.clear();//important
        time_stamp++;
        vector<particle> tmp_new_states;
        for (int i=0;i<n_particles;i++){
            particle state=states[i];
            float _x,_y,_dx,_dy,_width,_height,_dw,_dh;
            _dx=(state.x-state.x_p);
            _dy=(state.y-state.y_p);
            _dw=(state.width-state.width_p);
            _dh=(state.height-state.height_p);
            _x=MIN(MAX(cvRound(state.x+_dx+position_random_walk(generator)),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y+_dy+position_random_walk(generator)),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width+_dw+scale_random_walk(generator)),0),im_size.width);
            _height=MIN(MAX(cvRound(state.height+_dh+scale_random_walk(generator)),0),im_size.height);
            if((_x+_width)<im_size.width && _x>0 && 
                (_y+_height)<im_size.height && _y>0 && 
                _width<im_size.width && _height<im_size.height && 
                _width>0 && _height>0){
                state.x_p=state.x;
                state.y_p=state.y;
                state.width_p=state.width;
                state.height_p=state.height;       
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
            }
            else{
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

void discrete_particle_filter::draw_particles(Mat& image, Scalar color=Scalar(0,255,255)){
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

Rect discrete_particle_filter::estimate(Mat& image,bool draw=false){
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0;
    Rect estimate;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        double weight=weights[i];
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
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 1, LINE_AA );
        estimate=Rect(pt1.x,pt1.y,pt2.x,pt2.y);
    }
    return estimate;
}


void discrete_particle_filter::update(Mat& image)
{
    Mat grayImg;
    cvtColor(image, grayImg, CV_RGB2GRAY);
    haar.getFeatureValue(grayImg,sampleBox);         
    vector<double> tmp_weights; 
    //cout << reference_hist << endl; 
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi;
        particle state=states[i];
        if (state.width < 0 || state.width>image.cols){
          state.width = reference_roi.width;
        }
        if (state.height < 0 || state.height>image.rows){
          state.height = reference_roi.width;
        }
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox).clone();
        calc_hist_hsv(part_roi,part_hist);
        VectorXd counts;
        counts.resize(part_hist.total());
        double k=0.0;
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ ){
                double val=part_hist.at<float>(h, s);
                k+=val;
                counts[h*S_BINS+s] = (val!=0.0) ? val : eps;
            }
        double lambda=theta_y.array().sum();    
        double poisson_log_prior=k * log(lambda) - lgamma(k + 1.0) - lambda;    
        double prob=color_likekihood.log_likelihood(counts);
        double weight=weights[i]+prob;
        if(HOG){
            VectorXd count_hog;
            Scalar muTemp;
            Scalar sigmaTemp;
            count_hog.resize(haar.featureNum);
            for(int j=0;j<haar.featureNum;j++)
                count_hog[j]=haar.sampleFeatureValue.at<float>(i,j);    
            double prob_hog = hog_likelihood.log_likelihood(count_hog);
            weight+=prob_hog+poisson_log_prior;
        }
        tmp_weights.push_back(weight);
        part_hist.release();
        part_roi.release();
    }
    weights.swap(tmp_weights);
    tmp_weights=vector<double>();
    resample();
}

void discrete_particle_filter::resample(){
    vector<double> cumulative_sum(n_particles);
    vector<double> normalized_weights(n_particles);
    vector<double> new_weights(n_particles);
    vector<double> squared_normalized_weights(n_particles);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    float logsumexp=0.0;
    float max_value = *max_element(weights.begin(), weights.end());
    for (unsigned int i=0; i<weights.size(); i++) {
        new_weights[i]=weights[i]-max_value;
        logsumexp+=exp(weights[i]-max_value);
    }
    float norm_const=max_value+log(logsumexp);
    for (unsigned int i=0; i<weights.size(); i++) {
        normalized_weights[i] = exp(weights[i]-norm_const);
    }
    for (unsigned int i=0; i<weights.size(); i++) {
        squared_normalized_weights[i]=normalized_weights[i]*normalized_weights[i];
        if (i==0) {
            cumulative_sum[i] = normalized_weights[i];
        } else {
            cumulative_sum[i] = cumulative_sum[i-1] + normalized_weights[i];
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
            weights[i]=1.0f/n_particles;
        }
        states.swap(new_states);
    }
    else{
        weights.swap(new_weights);
    }
    cumulative_sum=vector<double>();
    normalized_weights=vector<double>();
    new_weights=vector<double>();
    squared_normalized_weights=vector<double>();
}

float discrete_particle_filter::getESS(){
    return ESS;
}

void discrete_particle_filter::update_model(VectorXd theta_x_new,VectorXd theta_y_new){
    theta_x=theta_x_new;
    theta_y=theta_y_new;
}


VectorXd discrete_particle_filter::get_dynamic_model(){
    return theta_x;
}

VectorXd discrete_particle_filter::get_observation_model(){
    return theta_y;
}
double discrete_particle_filter::getMarginalLikelihood(){
    return marginal_likelihood;
}
