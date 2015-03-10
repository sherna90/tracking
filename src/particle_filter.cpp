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
    for (int i=0;i<n_particles;i++){
        particle state=states[time_stamp][i];
        float _x,_y,_width,_height;
        _x=cvRound(state.x+state.dx+rng.gaussian(POS_STD));
        _y=cvRound(state.y+state.dy+rng.gaussian(POS_STD));
        _width=cvRound(state.width*state.scale);
        _height=cvRound(state.height*state.scale);
        if((_x+_width)<im_size.width && _x>=0 && (_y+_height)<im_size.height && _y>=0 && isless(getESS(),THRESHOLD)){
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
        states[time_stamp+1].push_back(state);
    }
    time_stamp++;
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
    //_x/=n_particles;
    //_y/=n_particles;
    //_width/=n_particles;
    //_height/=n_particles;   
    Point pt1,pt2;
    pt1.x=cvRound(_x);
    pt1.y=cvRound(_y);
    pt2.x=cvRound(_x+_width);
    pt2.y=cvRound(_y+_height);
    if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 1, LINE_AA );
    return Rect(pt1.x,pt1.y,cvRound(pt2.x-pt1.x),cvRound(pt2.y-pt1.y));
}


void particle_filter::update(Mat& image,Mat& reference_hist,Mat& reference_hog)
{
    weights[time_stamp].resize(n_particles);
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
            prob_color = 1.0f/sqrt(2.0f*SIGMA_COLOR)*exp(-LAMBDA_COLOR * (bc_color * bc_color) );
         if( bc_hog != 1.0f) // Clamp total mismatch to 0 likelihood
            prob_hog = 1.0f/sqrt(2.0f*SIGMA_SHAPE)*exp(-LAMBDA_SHAPE * (bc_hog * bc_hog));
        //weights[i]=weights[i]*(ALPHA*prob_color+(1-ALPHA)*prob_hog);
        weights[time_stamp].at(i)=weights[time_stamp-1][i]*prob_color*prob_hog;
    }
    resample();
}

void particle_filter::update(Mat& image,Mat& reference_hist)
{
    weights[time_stamp].resize(n_particles);
    static const float LAMBDA_COLOR = 0.5f*1.0f/(pow(SIGMA_COLOR,2.0f));
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        float bc_color = compareHist(reference_hist, part_hist, HISTCMP_BHATTACHARYYA);
        float prob = 0.0f;
        if(bc_color != 1.0f ) // Clamp total mismatch to 0 likelyhood
            prob = 1.0f/sqrt(2.0f*SIGMA_COLOR)*exp(-LAMBDA_COLOR * (bc_color * bc_color) );
        weights[time_stamp].at(i)=weights[time_stamp-1][i]*prob;
    }
    resample();
}

void particle_filter::update_dirichlet(Mat& image,Mat& reference_hist)
{
    weights[time_stamp].resize(n_particles);
    Eigen::VectorXd alpha,counts;
    alpha.setOnes(reference_hist.total());
    for(int h=0;h<H_BINS;h++)
        for( int s = 0; s < S_BINS; s++ )
        {
            alpha[h*S_BINS+s] = (double)reference_hist.at<float>(h, s);
        }
    dirichlet polya(alpha);
    //cout << alpha.transpose() << endl; 
    //polya.dirichlet_moment_match(alpha);
    for (int i=0;i<n_particles;i++){
        Mat part_hist,part_roi,part_hog;
        particle state=states[time_stamp][i];
        Rect boundingBox=Rect(cvRound(state.x),cvRound(state.y),cvRound(state.width),cvRound(state.height));
        part_roi=image(boundingBox);
        calc_hist_hsv(part_roi,part_hist);
        counts.setOnes(part_hist.total());
        for(int h=0;h<H_BINS;h++)
            for( int s = 0; s < S_BINS; s++ )
            {
                counts[h*S_BINS+s] = (double)part_hist.at<float>(h, s);
            }
        cout << "particle : " << i << ",time stamp : " << time_stamp<< endl;    
        cout << "histogram : " << counts.transpose() << endl; 
        cout << "alpha : " << alpha.transpose() << endl; 
        double prob = polya.log_likelihood(counts);
        weights[time_stamp].at(i)=weights[time_stamp-1][i]*exp(prob);
        cout << "log-likelihood : " << prob << endl; 
        cout << "--------------------------------" << endl; 

    }
    //resample();
}

void particle_filter::resample(){
    vector<float> cumulative_sum(n_particles);
    vector<float> normalized_weights(n_particles);
    vector<float> squared_normalized_weights(n_particles);
    Scalar s = sum(weights[time_stamp]);
    for (int i=0; i<weights[time_stamp].size(); i++) {
        normalized_weights[i] = weights[time_stamp][i] / s[0];
        squared_normalized_weights[i]=pow(normalized_weights[i],2.0f);
        if (i==0) {
            cumulative_sum[i] = normalized_weights[i];
        } else {
                cumulative_sum[i] = cumulative_sum[i-1] + normalized_weights[i];
            }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    ESS=1.0f/sum_squared_weights[0];
    if(isless(ESS/n_particles,THRESHOLD)){
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