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



void particle_filter::initialize(Rect roi,Size _im_size,Mat& _reference_hist,Mat& _reference_hog) {
    //weights[0].resize(n_particles);
    reference_hist=_reference_hist;
    reference_hog=_reference_hog;
    im_size=_im_size;
    smoothing_weights.resize(n_particles);
    vector<particle> tmp_states;
    vector<float> tmp_weights;
    for (int i=0;i<n_particles;i++){
        particle state;
        state.x=rng.uniform(0, im_size.width-roi.width);
        state.y=rng.uniform(0, im_size.height-roi.height);
        state.dx=rng.gaussian(VEL_STD);
        state.dy=rng.gaussian(VEL_STD);
        state.scale=1.f+rng.gaussian(SCALE_STD);
        tmp_states.push_back(state);
        tmp_weights.push_back(1.f/n_particles);
        ESS=0.0f;
        reference_roi=roi;
        state.width=reference_roi.width;
        state.height=reference_roi.height;      
    }
    states.push_back(tmp_states);
    weights.push_back(tmp_weights);
    color_lilekihood=Gaussian(0.0,SIGMA_COLOR);
    hog_likelihood=Gaussian(0.0,SIGMA_SHAPE);
    double eps= std::numeric_limits<double>::epsilon();
    Eigen::VectorXd alpha,alpha_hog;
    alpha.setOnes(reference_hist.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ ){
            double val=reference_hist.at<float>(h, s);
            alpha[h*S_BINS+s] = (val!=0.0) ? val : eps;
        }
    alpha_hog.setOnes(reference_hog.total());
    for(unsigned int g=0;g<reference_hog.total();g++){
        double val=reference_hog.at<float>(0,g);
        alpha_hog[g] = (val!=0.0) ? val : eps;
    }
    polya = dirichlet(alpha);
    alpha /=alpha.sum();
    discrete = Multinomial(alpha);  
    polya_hog = dirichlet(alpha_hog);
    alpha_hog /=alpha_hog.sum();
    discrete_hog = Multinomial(alpha_hog);  
    cout << "initialized!" << endl;  
    initialized=true;
}

void particle_filter::predict(){
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states;
        vector<particle> tmp_states=states.front();
        for (int i=0;i<n_particles;i++){
            particle state=tmp_states[i];
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
            tmp_new_states.push_back(state);
        }
        states.push_back(tmp_new_states);
    }
}

void particle_filter::smoother(int fixed_lag){
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
            //cout << "num frame :" << time_stamp << "; fixed-lag:" <<  k << endl;
    
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
    Rect estimate;
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
    if(pt2.x<im_size.width && pt1.x>=0 && pt2.y<im_size.height && pt1.y>=0){
        estimate=Rect(pt1.x,pt1.y,cvRound(pt2.x-pt1.x),cvRound(pt2.y-pt1.y));
    }
    else{
        cout << "oops!" << endl;
    }
    return estimate;
}

Rect particle_filter::smoothed_estimate(int fixed_lag){
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


void particle_filter::update(Mat& image,Mat& fgmask,bool hog=false)
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

void particle_filter::update_discrete(Mat& image,Mat& fgmask,bool dirichlet=false,bool hog = false){
    double lambda=polya.getAlpha().sum();
    vector<float> tmp_weights;
    double eps= std::numeric_limits<double>::epsilon();
    double prob = 0.0f;
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        Mat smoothed_mask = Mat(fgmask,boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        calc_hog(part_roi,part_hog);        
        Eigen::VectorXd counts,hog_counts;
        counts.setOnes(part_hist.total());
        double k=0.0;
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ )
            {
                double val=part_hist.at<float>(h, s);
                k+=val;             
                counts[h*S_BINS+s] = (val!=0.0) ? val : eps;
            }
        float poisson_log_prior=k * log(lambda) - lgamma(k + 1.0) - lambda;
        if(dirichlet) prob = polya.log_likelihood(counts)+poisson_log_prior;
        else prob=discrete.log_likelihood(counts)+poisson_log_prior;
        float weight=log(weights[time_stamp-1][i])+prob;
        if(hog){
            calc_hog(part_roi,part_hog);
            hog_counts.setOnes(part_hog.total());
            if(part_hog.size()==reference_hog.size()){
                for(unsigned int g=0;g<part_hog.total();g++){
                    double val=part_hog.at<float>(0, g);
                    hog_counts[g] = (val!=0.0) ? val : eps;
                }
                double prob_hog;
                if(dirichlet){
                    prob_hog = polya_hog.log_likelihood(hog_counts);
                }else{
                    prob_hog = discrete_hog.log_likelihood(hog_counts);
                }                
                weight*=prob_hog;
            }
        }
        //cout << weight << ";" << counts.transpose() << endl;
        tmp_weights.push_back(weight);
    }
    weights.push_back(tmp_weights);
    resample(true);
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
    for (unsigned int i=0; i<weights[time_stamp].size(); i++) {
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

void particle_filter::update_model(Mat& previous_frame,Mat& fgmask,Rect& smoothed_estimate){
    
    double eps= std::numeric_limits<double>::epsilon();
    Mat smoothed_hist;
    Mat smoothed_roi = Mat(previous_frame,smoothed_estimate);
    Mat smoothed_mask = Mat(fgmask,smoothed_estimate);
    calc_hist_hsv(smoothed_roi,smoothed_mask,smoothed_hist); 
    Eigen::VectorXd counts;
    counts.setOnes(smoothed_hist.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ )
        {
            double val=smoothed_hist.at<float>(h, s);            
            counts[h*S_BINS+s] = (val!=0.0) ? val : eps;
        }
    //double alpha=0.1;
    //discrete.addTheta(counts,alpha);
}