/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "../include/particle_filter.hpp"

const float POS_STD=0.1;
const float VEL_STD=0.1;
const float SCALE_STD=0.1;
const float  DT=1.0;
const float  SIGMA_COLOR=0.1;
const float  SIGMA_SHAPE=0.09;
const float  THRESHOLD=0.7;
const int  DIRICHLET_LIKELIHOOD=0;
const int MULTINOMIAL_LIKELIHOOD=1;
const int POISSON_LIKELIHOOD=2;
const int LIKELIHOOD=MULTINOMIAL_LIKELIHOOD;
const bool HOG=false;

particle_filter::particle_filter() {
}


particle_filter::~particle_filter() {
    states=vector<particle>();
    weights=vector<double>();
}

particle_filter::particle_filter(int _n_particles) {
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    //unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    //generator.seed(seed1);
    theta.resize(3);
    theta << POS_STD,VEL_STD,SCALE_STD;
    normal_distribution<double> position_random_walk(0.0,theta(0));
    normal_distribution<double> velocity_random_walk(0.0,theta(1));
    normal_distribution<double> scale_random_walk(0.0,theta(2));
    eps= std::numeric_limits<double>::epsilon();
}


bool particle_filter::is_initialized() {
    return initialized;
}

void particle_filter::reinitialize() {
    initialized=false;
}


void particle_filter::initialize(Mat& current_frame, Rect ground_truth) {
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
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width && 
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        Mat current_roi = Mat(current_frame,reference_roi);
        calc_hist_hsv(current_roi,reference_hist);
        marginal_likelihood=0.0;
        Eigen::VectorXd alpha,alpha_hog;
        color_lilekihood=Gaussian(0.0,SIGMA_COLOR);
        double weight=log(1.0/n_particles);
        for (int i=0;i<n_particles;i++){
            particle state;
            state.x=cvRound(reference_roi.x+position_random_walk(generator));
            state.y=cvRound(reference_roi.y+position_random_walk(generator));
            state.dx+=velocity_random_walk(generator);
            state.dy+=velocity_random_walk(generator);
            state.scale=1.0f+scale_random_walk(generator);
            states.push_back(state);
            weights.push_back(weight);
            ESS=0.0f;
            state.width=cvRound(right-left+state.scale);
            state.height=cvRound(bottom-top+state.scale);
        }
        alpha.setOnes(reference_hist.total());
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ ){
                double val=reference_hist.at<float>(h, s);
                gamma_distribution<double> color_prior(val,1.0);
                //alpha[h*S_BINS+s] = (val!=0.0) ? color_prior(generator) : eps;
                alpha[h*S_BINS+s] = (val!=0.0) ? val : eps;
            }
        polya = dirichlet(alpha);
        poisson = Poisson(alpha);
        alpha.normalize();
        discrete = Multinomial(alpha);
        if(HOG){
            calc_hog(current_roi,reference_hog);            
            hog_likelihood=Gaussian(0.0,SIGMA_SHAPE);
            alpha_hog.setOnes(reference_hog.total());
            for(unsigned int g=0;g<reference_hog.total();g++){
                double val=reference_hog.at<float>(0,g);
                gamma_distribution<double> hog_prior(val,1.0);
                alpha_hog[g] = (val!=0.0) ? hog_prior(generator) : eps;
            }
            polya_hog = dirichlet(alpha_hog);
            alpha_hog.normalize();
            discrete_hog = Multinomial(alpha_hog);
        }
        initialized=true;
    }
}

void particle_filter::predict(){
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states;
        uniform_real_distribution<double> unif_rnd(0.0,1.0);
        uniform_int_distribution<int> unif_width(reference_roi.width,(int)(im_size.width-reference_roi.width-1));
        uniform_int_distribution<int> unif_height(reference_roi.height,(int)(im_size.height-reference_roi.height-1));                  
        for (int i=0;i<n_particles;i++){
            particle state=states[i];
            float _x,_y,_dx,_dy,_width,_height;
            _dx=state.dx;//+velocity_random_walk(generator);
            _dy=state.dy;//+velocity_random_walk(generator);
            _x=MAX(cvRound(state.x+_dx+position_random_walk(generator)),0);
            _y=MAX(cvRound(state.y+_dy+position_random_walk(generator)),0);
            _width=MAX(cvRound(state.width),0);
            _height=MAX(cvRound(state.height),0);
            
            if((_x+_width)<im_size.width && _x>0 && 
                (_y+_height)<im_size.height && _y>0 && 
                _width<im_size.width && _height<im_size.height 
                && isless(ESS/n_particles,(float)THRESHOLD)){
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.dx=_dx;
                state.dy=_dy;
                state.scale+=scale_random_walk(generator);
            }
            else{
                state.dx=velocity_random_walk(generator);
                state.dy=velocity_random_walk(generator);
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
                double u=unif_rnd(generator);
                if(u<0.5){
                state.x=cvRound(reference_roi.x+position_random_walk(generator));
                state.y=cvRound(reference_roi.y+position_random_walk(generator));
                }
                else{
                    double val_x=unif_width(generator);
                    state.x= (val_x>0 && val_x <(im_size.width-state.width)) ? val_x : reference_roi.x;
                    double val_y=unif_height(generator);
                    state.y= (val_y>0 && val_y <(im_size.height-state.height)) ? val_y : reference_roi.y;
                }
                state.scale=1.f+scale_random_walk(generator);
            }
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
            tmp_new_states.push_back(state);
        }
        states.swap(tmp_new_states);
        tmp_new_states = vector<particle>();
    }
}


void particle_filter::draw_particles(Mat& image){
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        Point pt1,pt2;
        pt1.x=cvRound(state.x);
        pt1.y=cvRound(state.y);
        pt2.x=cvRound(state.x+state.width);
        pt2.y=cvRound(state.y+state.height);
        rectangle( image, pt1,pt2, Scalar(0,255,255), 1, LINE_AA );
    }
}

Rect particle_filter::estimate(Mat& image,bool draw=false){
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0;
    Rect estimate;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        double weight=exp(weights[i]);
        _x+=(weight*state.x > 0 && weight*state.x < im_size.width) ? weight*state.x : float(1/n_particles)*reference_roi.x; ;
        _y+=(weight*state.y > 0 && weight*state.y < im_size.height) ? weight*state.y : float(1/n_particles)*reference_roi.y; ;
        _width+=(weight*state.width > 0 && weight*state.width < im_size.width) ? weight*state.width : float(1/n_particles)*reference_roi.width; 
        _height+=(weight*state.height > 0 && weight*state.height < im_size.height) ? weight*state.height : float(1/n_particles)*reference_roi.height; 
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
        estimate=Rect(pt1.x,pt1.y,_width,_height);
    }
    return estimate;
}


void particle_filter::update(Mat& image)
{
    vector<double> tmp_weights;
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[i];
        if (state.width < 0){
          state.width = reference_roi.width;
        }
        if (state.height < 0){
          state.height = reference_roi.width;
        }
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        double bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        double prob = 0.0f;
        if(bc_color != 1.0f ){
            prob = color_lilekihood.log_likelihood(bc_color);
        }
        double weight=weights[i]+prob;
        if(HOG){
            calc_hog(part_roi,part_hog);
            if(part_hog.size()==reference_hog.size()){
                double bc_hog = compareHist(reference_hog, part_hog, HISTCMP_BHATTACHARYYA);
                double prob_hog = hog_likelihood.log_likelihood(bc_hog);
                weight+=prob_hog;
            }
        }
        tmp_weights.push_back(weight);
    }
    weights.swap(tmp_weights);
    tmp_weights=vector<double>();
    resample();
}

void particle_filter::update_discrete(Mat& image){
    //double lambda=polya.getAlpha().sum();
    vector<double> tmp_weights;
    double prob = 0.0f;
    double weight=-10.0;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        if (state.width < 0 || state.width > im_size.width){
          state.width = reference_roi.width;
        }
        if (state.height < 0 ||  state.height > im_size.height){
          state.height = reference_roi.width;
        }
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        if(boundingBox.x>0 && boundingBox.width>0 && (boundingBox.x+boundingBox.width)<im_size.width && 
            boundingBox.y>0 && boundingBox.height>0 && (boundingBox.y+boundingBox.height)<im_size.height){
            Mat part_hist,part_roi,part_hog;
            part_roi=image(boundingBox);
            calc_hist_hsv(part_roi,part_hist);
            VectorXd counts;
            counts.setOnes(part_hist.total());
            //double k=0.0;
            for(int h=0;h<H_BINS;h++)
                for( int s = 0; s < S_BINS; s++ )
                {
                    double val=part_hist.at<float>(h, s);
                    counts[h*S_BINS+s] = (val!=0.0) ? val : eps;
                }
            //double poisson_log_prior=k * log(lambda) - lgamma(k + 1.0) - lambda;
            if(LIKELIHOOD==DIRICHLET_LIKELIHOOD) prob = polya.log_likelihood(counts);
            else if(LIKELIHOOD==MULTINOMIAL_LIKELIHOOD) prob=discrete.log_likelihood(counts);        
    	    else prob=poisson.log_likelihood(counts);
            weight=weights[i]+prob;
            if(HOG){
                VectorXd hog_counts;
                calc_hog(part_roi,part_hog);
                hog_counts.setOnes(part_hog.total());
                if(part_hog.size()==reference_hog.size()){
                    //for(unsigned int g=0;g<part_hog.total();g++){
                    //    double val=part_hog.at<float>(0, g);
                    //    hog_counts[g] = (val!=0.0) ? val : eps;
                    //}
                    double bc_hog = compareHist(reference_hog, part_hog, HISTCMP_BHATTACHARYYA);
                    double prob_hog = hog_likelihood.log_likelihood(bc_hog);
                    weight+=prob_hog;
                }
            }
        }
        tmp_weights.push_back(weight);
    }
    weights.swap(tmp_weights);
    tmp_weights=vector<double>();
    resample();
}

void particle_filter::resample(){
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
        squared_normalized_weights[i]=pow(normalized_weights[i],2.0f);
        if (i==0) {
            cumulative_sum[i] = normalized_weights[i];
        } else {
            cumulative_sum[i] = cumulative_sum[i-1] + normalized_weights[i];
        }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    marginal_likelihood+=norm_const-log(n_particles); 
    ESS=1.0f/sum_squared_weights[0];
    if(isless(ESS/n_particles,(float)THRESHOLD)){
        vector<particle> new_states;
        for (int i=0; i<n_particles; i++) {
            double uni_rand = unif_rnd(generator);
            vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle state=states[ipos];
            new_states.push_back(state);
            weights[i]=log(1.0f/n_particles);
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

float particle_filter::getESS(){
    return ESS/n_particles;
}

void particle_filter::update_model(VectorXd theta_new,VectorXd alpha_new){
    //double alpha=0.1;
    poisson.setLambda(alpha_new);
    alpha_new.normalize();
    discrete.setTheta(alpha_new);
    theta=theta_new;
}

VectorXd particle_filter::get_discrete_model(){
    //double alpha=0.1;
    return discrete.getTheta();

}

VectorXd particle_filter::get_continuous_model(){
    return theta;

}

double particle_filter::getMarginalLikelihood(){
    return marginal_likelihood;
}
