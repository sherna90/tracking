/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
 #include "../include/particle_filter.hpp"


particle_filter::particle_filter(int _n_particles) {
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    //rng(0xFFFFFFFF);
}


bool particle_filter::is_initialized() {
    return initialized;
}

void particle_filter::initialize(Rect roi,Size im_size) {
    weights[0].resize(n_particles);
    smoothing_weights.resize(n_particles);
    for (int i=0;i<n_particles;i++){
        particle state;
        state.x=rng.uniform(0, im_size.width-roi.width);
        state.y=rng.uniform(0, im_size.height-roi.height);
        state.dx=rng.gaussian(VEL_STD);
        state.dy=rng.gaussian(VEL_STD);
        state.scale=1.f+rng.gaussian(SCALE_STD);
        states[0].push_back(state);
        weights[0].at(i)=1.f/n_particles;
        ESS=0.0f;
        reference_roi=roi;
        state.width=reference_roi.width;
        state.height=reference_roi.height;
        
    }
    initialized=true;
}

void particle_filter::predict(Size im_size){
    if(initialized==true){
        time_stamp++;
        for (int i=0;i<n_particles;i++){
            particle state=states[time_stamp-1][i];
            float _x,_y,_width,_height;
            _x=cvRound(state.x+state.dx+rng.gaussian(POS_STD));
            _y=cvRound(state.y+state.dy+rng.gaussian(POS_STD));
            _width=cvRound(state.width*state.scale);
            _height=cvRound(state.height*state.scale);
            if((_x+_width)<im_size.width && _x>=0 && (_y+_height)<im_size.height && _y>=0 && isless(getESS(),(float)THRESHOLD)){
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.dx+=rng.gaussian(VEL_STD);
                state.dy+=rng.gaussian(VEL_STD);
                state.scale+=rng.gaussian(SCALE_STD);
            }
            else{
                state.dx=rng.gaussian(VEL_STD);
                state.dy=rng.gaussian(VEL_STD);
                state.width=reference_roi.width;
                state.height=reference_roi.height;
                state.x=rng.uniform(0, (int)(im_size.width-state.width));
                state.y=rng.uniform(0, (int)(im_size.height-state.height));
                state.scale=1.f+rng.gaussian(SCALE_STD);
            }
            states[time_stamp].push_back(state);
        }
    }
}

void particle_filter::smoother(int fixed_lag){
    smoothing_weights=weights[time_stamp];
    vector<float> log_backward_probability(n_particles,0.0f);
    vector<float> normalized_weights(n_particles,0.0f);
    vector<float> sum_weights(n_particles,0.0f);
    static const float LAMBDA_POS = 0.5f*1.0f/(pow(POS_STD,2.0f));
    if(fixed_lag<time_stamp){
        for(unsigned int k=time_stamp;k>(time_stamp-fixed_lag);k--){
            for (unsigned int j=0;j<n_particles;j++){
                particle state=states[k][j];
                for (unsigned int l=0;l<n_particles;l++){
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
           
            for (unsigned int i=0;i<n_particles;i++){
                particle past_state=states[k-1][i];
                float backward_probability=0.0f;
                for (unsigned int j=0;j<n_particles;j++){
                    particle state=states[k][j];
                    float sum=LAMBDA_POS*pow(state.x-past_state.x-past_state.dx,2.0);
                    sum+=LAMBDA_POS*pow(state.y-past_state.y-past_state.dy,2.0);
                    float log_prob=log(weights[k-1].at(i))-log(2.0f*M_PI)-2.0f*log(POS_STD)-sum;
                    backward_probability+=exp(log_prob-sum_weights[j]);
                }
                smoothing_weights.at(i) = weights[k-1].at(i)*backward_probability;
                //smoothing_weights.at(i) = weights[k-1].at(i);
       
            }
            //cout << "Lag:" << k << endl;
        }
    }
}


void particle_filter::draw_particles(Mat& image){
    for (int i=0;i<n_particles;i++){
        particle state=states[time_stamp][i];
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
    for (int i=0;i<n_particles;i++){
        particle state=states[time_stamp][i];
        _x+=weights[time_stamp][i]*state.x;
        _y+=weights[time_stamp][i]*state.y;
        _width+=weights[time_stamp][i]*state.width;
        _height+=weights[time_stamp][i]*state.height;
    }  
    Point pt1,pt2;
    pt1.x=cvRound(_x);
    pt1.y=cvRound(_y);
    pt2.x=cvRound(_x+_width);
    pt2.y=cvRound(_y+_height);
    if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 1, LINE_AA );
    return Rect(pt1.x,pt1.y,cvRound(pt2.x-pt1.x),cvRound(pt2.y-pt1.y));
}

Rect particle_filter::smoothed_estimate(Mat& image,int fixed_lag,bool draw=false){
    //smoothing_weights=weights[time_stamp];
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0;
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
    if(draw) rectangle( image, pt1,pt2, Scalar(0,255,255), 1, LINE_AA );
    return Rect(pt1.x,pt1.y,cvRound(pt2.x-pt1.x),cvRound(pt2.y-pt1.y));
}

void particle_filter::update(Mat& image,Mat& reference_hist,Mat& reference_hog)
{
    weights[time_stamp].resize(n_particles,0.0f);
    static const float LAMBDA_COLOR = 0.5f*1.0f/(pow(SIGMA_COLOR,2.0f));
    static const float LAMBDA_SHAPE = 0.5f*1.0f/(pow(SIGMA_SHAPE,2.0f));
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        calc_hog(part_roi,part_hog);
        float bc_hog = 0.0f; 
        if(part_hog.size()==reference_hog.size())
            bc_hog = compareHist(reference_hog, part_hog, HISTCMP_BHATTACHARYYA);
        float bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        float prob_color = 0.0f;
        float prob_hog = 0.0f;
        if(bc_color != 1.0f) // Clamp total mismatch to 0 likelihood
            prob_color = (1.0f/sqrt(2.0f*M_PI*pow(SIGMA_COLOR,2.0f)))*exp(-LAMBDA_COLOR * (bc_color * bc_color) );
         if( bc_hog != 1.0f) // Clamp total mismatch to 0 likelihood
            prob_hog = (1.0f/sqrt(2.0f*M_PI*pow(SIGMA_SHAPE,2.0f)))*exp(-LAMBDA_SHAPE * (bc_hog * bc_hog));
        //weights[i]=weights[i]*(ALPHA*prob_color+(1-ALPHA)*prob_hog);
        weights[time_stamp].at(i)=weights[time_stamp-1][i]*prob_color*prob_hog;
    }
    resample(false);
}

void particle_filter::update(Mat& image,Mat& reference_hist)
{
    weights[time_stamp].resize(n_particles,0.0f);
    static const float LAMBDA_COLOR = 0.5f*1.0f/(pow(SIGMA_COLOR,2.0f));
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        float bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        float prob = 0.0f;
        if(bc_color != 1.0f ) // Clamp total mismatch to 0 likelihood
            prob = (1.0f/sqrt(2.0f*pow(SIGMA_COLOR,2.0f)))*exp(-LAMBDA_COLOR * (bc_color * bc_color) );
        weights[time_stamp].at(i)=weights[time_stamp-1][i]*prob;
    }
    resample(false);
}

void particle_filter::update_dirichlet(Mat& image,Mat& reference_hist){
    weights[time_stamp].resize(n_particles,0.0f);
    double lambda=0.0;
    static const float LAMBDA_COLOR = 0.5f/(pow(SIGMA_COLOR,2.0f));
    Eigen::VectorXd alpha,counts;
    alpha.setOnes(reference_hist.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ )
        {
            alpha[h*S_BINS+s] = (reference_hist.at<float>(h, s)>0.0f)?reference_hist.at<float>(h, s):DBL_EPSILON;
            lambda+=reference_hist.at<float>(h, s);
        }
    //cout << "dirichlet precision : " << alpha.sum()<< endl; 
    //cout << "H = " << reference_hist << endl;  
    dirichlet polya(alpha);
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        counts.setOnes(part_hist.total());
        double k=0.0;
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ )
            {
                k+=part_hist.at<float>(h, s);
                counts[h*S_BINS+s] = part_hist.at<float>(h, s);
            }
        double prob = polya.log_likelihood(counts)+k * log(lambda) - lgamma(k + 1.0) - lambda;
        weights[time_stamp].at(i)=log(weights[time_stamp-1][i])+prob; 
        float bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        //cout << "log-likelihood DCM: " << prob << endl;
        //cout << "sample = " << part_hist << ", ";   
        //cout  << prob << ",";
        if(bc_color != 1.0f ) {      
            prob = log(1.0f/sqrt(2.0f*SIGMA_COLOR))-(LAMBDA_COLOR * (bc_color * bc_color) ); 
        }
        //cout <<  prob << endl; 
        //cout << "log-likelihood bhattarchaya: " << prob << endl;
        //cout << "log weight : " << weights[time_stamp].at(i) << endl;
        //cout << "--------------------------------" << endl; 

    }
    resample(true);
    //exit(1);
}

void particle_filter::update_dirichlet(Mat& image,Mat& reference_hist,Mat& reference_hog){
    weights[time_stamp].resize(n_particles,0.0f);
    double lambda=0.0;
    Eigen::VectorXd alpha,counts,alpha_h,counts_h;
    alpha.setOnes(reference_hist.total());
    alpha_h.setOnes(reference_hog.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ )
        {
            alpha[h*S_BINS+s] = (reference_hist.at<float>(h, s)>0.0f)?reference_hist.at<float>(h, s):DBL_EPSILON;
            lambda+=reference_hist.at<float>(h, s);
        }
    for( int g = 0; g < reference_hog.total(); g++ ){
            alpha_h[g] = (reference_hog.at<float>(0, g)>0.0f)?reference_hog.at<float>(0, g):DBL_EPSILON;
        }
    dirichlet polya(alpha);
    dirichlet polya_h(alpha_h);
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        calc_hog(part_roi,part_hog);
        counts.setOnes(part_hist.total());
        double k=0.0;
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ )
            {
                k+=part_hist.at<float>(h, s);
                counts[h*S_BINS+s] = part_hist.at<float>(h, s);
            }
        counts_h.setOnes(part_hog.total());
        for( int g = 0; g < part_hog.total(); g++ ){
            counts_h[g] = part_hog.at<float>(0, g);
        }
        double prob = polya_h.log_likelihood(counts_h);
        //cout << "sample = " << part_hog << ", "; 
        weights[time_stamp].at(i)=log(weights[time_stamp-1][i])+prob;  

    }
    resample(true);
    //exit(1);
}

void particle_filter::resample(bool log_scale=false){
    vector<float> cumulative_sum(n_particles);
    vector<float> normalized_weights(n_particles);
    vector<float> squared_normalized_weights(n_particles);
    float logsumexp=0.0;
    if(log_scale){
        float max_value = *max_element(weights[time_stamp].begin(), weights[time_stamp].end());
        for (unsigned int i=0; i<weights[time_stamp].size(); i++) {
            logsumexp+=exp(weights[time_stamp][i]-max_value);
        }
        logsumexp=max_value+log(logsumexp);
    }
    Scalar s = sum(weights[time_stamp]);
    for (int i=0; i<weights[time_stamp].size(); i++) {
        if(log_scale){
            normalized_weights[i] = exp(weights[time_stamp][i]-logsumexp); 
            }
        else{
            normalized_weights[i] = weights[time_stamp][i] / s[0];
        }
    }
    for (unsigned int i=0; i<weights[time_stamp].size(); i++) {
        squared_normalized_weights[i]=pow(normalized_weights[i],2.0f);
        if (i==0) {
            cumulative_sum[i] = normalized_weights[i];
        } else {
                cumulative_sum[i] = cumulative_sum[i-1] + normalized_weights[i];
            }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    ESS=1.0f/sum_squared_weights[0];
    if(isless(ESS/n_particles,(float)THRESHOLD)){
        vector<particle> new_states;
        for (int i=0; i<n_particles; i++) {
            float uni_rand = rng.uniform(0.0f,1.0f);
            vector<float>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle state=states[time_stamp][ipos];
            new_states.push_back(state);
            weights[time_stamp][i]=1.0f/n_particles;
        }
        states[time_stamp].swap(new_states);
    }
    else{
        weights[time_stamp].swap(normalized_weights);
    }
}

float particle_filter::getESS(){
    return ESS/n_particles;
}