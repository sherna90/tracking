/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=1.0;
const float SCALE_STD=1.0;
const float DT=1.0;


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
    vector<Rect> negativeBox;
    states.clear();
    weights.clear();
    estimates.clear();
    sampleBox.clear();
    estimates.push_back(ground_truth);
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
        }
        this->detector.init(GROUP_THRESHOLD, HIT_THRESHOLD, this->reference_roi);
        this->detector.train(current_frame_copy, this->reference_roi);
        this->initialized = true;
        cout << "initialized!!!" << endl;
        initialized=true;    
    }

}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    sampleBox.clear();//important
    //cout << "predicted particles!" <<endl;
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states(n_particles);
        for (int i = 0; i < n_particles; i++){
            particle state = states[i];
            float _x, _y, _width, _height;
            float _dx = position_random_x(generator);
            float _dy = position_random_y(generator);
            //float _dw=scale_random_width(generator);
            //float _dh=scale_random_height(generator);
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
                && _width>0 && _height>0 ){
                state.x_p=state.x;
                state.y_p=state.y;
                state.x=cvRound(2*_x-state.x_p+_dx);
                state.y=cvRound(2*_y-state.y_p+_dy);
                state.width_p=state.width;
                state.height_p=state.height;       
                state.width=cvRound(2*_width-state.width_p);
                state.height=cvRound(2*_height-state.height_p);
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
            box.x=MIN(MAX(cvRound(box.x),0),im_size.width);
            box.y=MIN(MAX(cvRound(box.y),0),im_size.height);
            box.width=MIN(MAX(cvRound(box.width),0),im_size.width-box.x);
            box.height=MIN(MAX(cvRound(box.height),0),im_size.height-box.y);
            sampleBox.push_back(box);
            
            //cout << "box " << box.height << " " << box.width << endl;

            //cout << "reference " << reference_roi << endl;
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height <<",weight:" << weights[i] << endl;
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
    Mat current_frame_copy;
    image.copyTo(current_frame_copy);

    int left = MAX(this->reference_roi.x, 1);
    int top = MAX(this->reference_roi.y, 1);
    int right = MIN(this->reference_roi.x + this->reference_roi.width, image.cols - 1);
    int bottom = MIN(this->reference_roi.y + this->reference_roi.height, image.rows - 1);
    Rect update_roi = Rect(left, top, right - left, bottom - top);
    vector<Rect> samples;
    for (size_t i = 0; i < this->states.size(); ++i){
            particle state = this->states[i];
            Rect current_state=Rect(state.x, state.y, state.width, state.height);
            samples.push_back(current_state);
    }
    this->dppResults = this->detector.detect(current_frame_copy,samples);


    //weights.swap(tmp_weights);
    tmp_weights.clear();
    resample();

}

float particle_filter::resample(){
    vector<float> cumulative_sum(n_particles);
    vector<float> normalized_weights(n_particles);
    vector<float> squared_normalized_weights(n_particles);
    uniform_real_distribution<float> unif_rnd(0.0,1.0); 
    float max_value = *max_element(weights.begin(), weights.end());
    float logsumexp=0.0f;
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = exp(weights.at(i)-max_value);
        logsumexp+=normalized_weights.at(i);
    }
    Scalar sum_weights=sum(normalized_weights);
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = normalized_weights.at(i)/sum_weights[0];
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
    marginal_likelihood+=max_value+log(sum_weights[0])-log(n_particles); 
    ESS=1/sum_squared_weights[0]/n_particles;
    //cout  << "ESS :" << ESS << ",marginal_likelihood :" << marginal_likelihood <<  endl;
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
        //weights.swap(log_normalized_weights);
    }
    cumulative_sum.clear();
    normalized_weights.clear();
    squared_normalized_weights.clear();
    return marginal_likelihood;
}

float particle_filter::getESS(){
    return ESS;
}

void particle_filter::update_model(vector<VectorXd> theta_x_new){
    theta_x.clear();
    VectorXd theta_x_pos=theta_x_new.at(0);
    VectorXd theta_x_scale=theta_x_new.at(1);
    theta_x.push_back(theta_x_pos);
    theta_x.push_back(theta_x_scale);
}

void particle_filter::update_model(Mat& current_frame,vector<Rect> positive_examples,vector<Rect> negative_examples){
    Mat grayImg;
    cvtColor(current_frame, grayImg, CV_RGB2GRAY);
    if(LOGISTIC_REGRESSION){
        VectorXd labels(positive_examples.size()+negative_examples.size());
        labels << VectorXd::Ones(positive_examples.size()), VectorXd::Constant(negative_examples.size(),-1.0);
        if(HAAR_FEATURE){
            MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
            haar.getFeatureValue(grayImg,positive_examples);
            cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
            haar.getFeatureValue(grayImg,negative_examples);
            cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
            MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
            eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                            eigen_sample_negative_feature_value;
            eigen_sample_feature_value.transposeInPlace();
            hamiltonian_monte_carlo.setData(eigen_sample_feature_value, labels);
        }

        if(LBP_FEATURE){
            local_binary_pattern.init(grayImg, positive_examples);
            local_binary_pattern.getFeatureValue(grayImg, negative_examples, false);
            MatrixXd eigen_sample_feature_value(local_binary_pattern.sampleFeatureValue.rows() +
            local_binary_pattern.negativeFeatureValue.rows(), local_binary_pattern.sampleFeatureValue.cols());
            eigen_sample_feature_value << local_binary_pattern.sampleFeatureValue,
                                          local_binary_pattern.negativeFeatureValue;
            hamiltonian_monte_carlo.setData(eigen_sample_feature_value, labels);
        }

        if(MB_LBP_FEATURE){
            multiblock_local_binary_patterns = MultiScaleBlockLBP(3,59,2,true,false,3,3);
            multiblock_local_binary_patterns.init(grayImg, positive_examples);
            multiblock_local_binary_patterns.getFeatureValue(grayImg, negative_examples, false);
            MatrixXd eigen_sample_feature_value(multiblock_local_binary_patterns.sampleFeatureValue.rows() +
                multiblock_local_binary_patterns.negativeFeatureValue.rows(), multiblock_local_binary_patterns.sampleFeatureValue.cols());
            eigen_sample_feature_value << multiblock_local_binary_patterns.sampleFeatureValue,
                                          multiblock_local_binary_patterns.negativeFeatureValue;
            hamiltonian_monte_carlo.setData(eigen_sample_feature_value, labels);
        }

        if(HOG_FEATURE){
            MatrixXd hog_descriptors(0, 3780);
            VectorXd hist;
            for (unsigned int i = 0; i < positive_examples.size(); ++i)
            {
                Mat subImage = grayImg(positive_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }

            for (unsigned int i = 0; i < negative_examples.size(); ++i)
            {
                Mat subImage = grayImg(negative_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }
            hamiltonian_monte_carlo.setData(hog_descriptors, labels);
        }
        hamiltonian_monte_carlo.run(10,1e-2,10);
    }
    if(GAUSSIAN_NAIVEBAYES){
        VectorXd labels(positive_examples.size()+negative_examples.size());
        labels << VectorXd::Ones(positive_examples.size()), VectorXd::Zero(negative_examples.size());
        double learning_rate = 0.9;
        if(HAAR_FEATURE){
            haar.init(grayImg,reference_roi,positive_examples);
            MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
            cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
            haar.getFeatureValue(grayImg,negative_examples);
            cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
            MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
            eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                            eigen_sample_negative_feature_value;
            eigen_sample_feature_value.transposeInPlace();
            gaussian_naivebayes.partial_fit(eigen_sample_feature_value, labels, learning_rate);
        }
        if(LBP_FEATURE){
            local_binary_pattern.init(grayImg, positive_examples);
            local_binary_pattern.getFeatureValue(grayImg, negative_examples, false);
            MatrixXd eigen_sample_feature_value(local_binary_pattern.sampleFeatureValue.rows() +
            local_binary_pattern.negativeFeatureValue.rows(), local_binary_pattern.sampleFeatureValue.cols());
            eigen_sample_feature_value << local_binary_pattern.sampleFeatureValue,
                                          local_binary_pattern.negativeFeatureValue;
            gaussian_naivebayes.partial_fit(eigen_sample_feature_value, labels, learning_rate);
        }
        if(MB_LBP_FEATURE){
            multiblock_local_binary_patterns = MultiScaleBlockLBP(3,59,2,true,false,3,3);
            multiblock_local_binary_patterns.init(grayImg, positive_examples);
            multiblock_local_binary_patterns.getFeatureValue(grayImg, negative_examples, false);
            MatrixXd eigen_sample_feature_value(multiblock_local_binary_patterns.sampleFeatureValue.rows() +
                multiblock_local_binary_patterns.negativeFeatureValue.rows(), multiblock_local_binary_patterns.sampleFeatureValue.cols());
            eigen_sample_feature_value << multiblock_local_binary_patterns.sampleFeatureValue,
                                          multiblock_local_binary_patterns.negativeFeatureValue;
            gaussian_naivebayes.partial_fit(eigen_sample_feature_value, labels, learning_rate);
        }
        if(HOG_FEATURE){
            MatrixXd hog_descriptors(0, 3780);
            VectorXd hist;
            for (unsigned int i = 0; i < positive_examples.size(); ++i)
            {
                Mat subImage = grayImg(positive_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }

            for (unsigned int i = 0; i < negative_examples.size(); ++i)
            {
                Mat subImage = grayImg(negative_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }
            gaussian_naivebayes.partial_fit(hog_descriptors, labels, learning_rate);
        }

    }

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

particle particle_filter::update_state(particle state, Mat& image){
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
    return state;
}
