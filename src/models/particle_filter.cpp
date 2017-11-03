/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=3.0;
const float SCALE_STD=0.1;
const float DT=1.0;
const float THRESHOLD=1.0;
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
    normal_distribution<double> position_random_x(0.0, this->theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0, this->theta_x.at(0)(1));
    this->frame_size=current_frame.size();
    normal_distribution<double> negative_random_pos(0.0,20.0);

    this->states.clear();
    this->weights.clear();
    //this->weights = VectorXd::Ones(this->n_particles) * log(1.0/(this->n_particles));
    double weight = 1.0f/this->n_particles;

    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    this->reference_roi = Rect(left, top, right - left, bottom - top);
    if ( (this->reference_roi.width > 0)
        && ((this->reference_roi.x + this->reference_roi.width) < frame_size.width)
        && (this->reference_roi.height > 0)
        && ((this->reference_roi.y + this->reference_roi.height) < frame_size.height) )
    {
        for (int i = 0; i < this->n_particles; ++i)
        {
            particle state;
            float _x, _y, _width, _height;
            float _dx = position_random_x(this->generator);
            float _dy = position_random_y(this->generator);

            _x = MIN(MAX(cvRound(this->reference_roi.x + _dx), 0), this->frame_size.width);
            _y = MIN(MAX(cvRound(this->reference_roi.y + _dy), 0), this->frame_size.height);
            _width = MIN(MAX(cvRound(this->reference_roi.width), 10.0), this->frame_size.width);
            _height = MIN(MAX(cvRound(this->reference_roi.height), 10.0), this->frame_size.height);

            if ( ((_x + _width) < this->frame_size.width)
                && (_x > 0)
                && ((_y + _height) < this->frame_size.height)
                && (_y > 0)
                && (_width < this->frame_size.width)
                && (_height < this->frame_size.height)
                && (_width > 0)
                && (_height > 0) )
            {
                state.x_p = this->reference_roi.x;
                state.y_p = this->reference_roi.y;
                state.width_p = this->reference_roi.width;
                state.height_p = this->reference_roi.height;
                state.scale_p = state.scale;

                state.x = _x;
                state.y = _y;
                state.width = _width;
                state.height = _height;
                state.scale = 1.0;
            }
            else
            {
                state.x_p = this->reference_roi.x;
                state.y_p = this->reference_roi.y;
                state.width_p = cvRound(this->reference_roi.width);
                state.height_p = cvRound(this->reference_roi.height);

                state.x = this->reference_roi.x;
                state.y = this->reference_roi.y;
                state.width = cvRound(this->reference_roi.width);
                state.height = cvRound(this->reference_roi.height);
                state.scale = 1.0;
            }
            //cout << "x: " << state.x << "\ty: " << state.y << "\twidth: " << state.width << "\theight: " << state.height << "\tweight: " << weight << endl;
            this->states.push_back(state);
            this->weights.push_back(weight);
        }

    Mat current_frame_copy;
    current_frame.copyTo(current_frame_copy);
    this->detector.init(0.0,0.0, this->reference_roi);
    this->detector.train(current_frame_copy, this->reference_roi);
    this->initialized = true;
    }
}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states(n_particles);
        for (int i = 0; i < n_particles; i++){
            particle state = states[i];
            float _x, _y, _width, _height;
            float _dx = position_random_x(generator);
            float _dy = position_random_y(generator);
            float _dw=scale_random_width(generator);
            float _dh=scale_random_height(generator);
            _x = MIN(MAX(cvRound(state.x + _dx), 0), this->frame_size.width);
            _y = MIN(MAX(cvRound(state.y + _dy), 0), this->frame_size.height);
            _width = MIN(MAX(cvRound(state.width+_dw), 0), this->frame_size.width);
            _height = MIN(MAX(cvRound(state.height+_dh), 0), this->frame_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),this->frame_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),this->frame_size.height);
            if( (_x+_width)<this->frame_size.width 
                && _x>0 
                && (_y+_height)<this->frame_size.height 
                && _y>0 
                && _width<this->frame_size.width 
                && _height<this->frame_size.height 
                && _width>0 && _height>0 ){
                state.x_p=state.x;
                state.y_p=state.y;
                state.x=cvRound(_x+_dx);
                state.y=cvRound(_y+_dy);
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
            box.x=MIN(MAX(cvRound(box.x),0),this->frame_size.width);
            box.y=MIN(MAX(cvRound(box.y),0),this->frame_size.height);
            box.width=MIN(MAX(cvRound(box.width),0),this->frame_size.width-box.x);
            box.height=MIN(MAX(cvRound(box.height),0),this->frame_size.height-box.y);
            
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
        if(state.x>0 && state.x<this->frame_size.width 
            && state.y>0  && state.y<this->frame_size.height 
            && state.width>0 && state.width<this->frame_size.height 
            && state.height>0 && state.height<this->frame_size.height){
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
    if(pt2.x<this->frame_size.width && pt1.x>=0 && pt2.y<this->frame_size.height && pt1.y>=0){
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
        estimate=Rect(pt1.x,pt1.y,_width,_height);
        this->reference_roi=estimate;
    }
    //cout << " estimate x:" << estimate.x << ",y:" << estimate.y <<",w:" << estimate.width <<",h:" << estimate.height << endl;
    return estimate;

}

void particle_filter::update(Mat& image)
{
    Mat current_frame;
    image.copyTo(current_frame);
    int left = MAX(this->reference_roi.x, 1);
    int top = MAX(this->reference_roi.y, 1);
    int right = MIN(this->reference_roi.x + this->reference_roi.width, image.cols - 1);
    int bottom = MIN(this->reference_roi.y + this->reference_roi.height, image.rows - 1);
    Rect update_roi = Rect(left, top, right - left, bottom - top);
    vector<Rect> samples;
    for (size_t i = 0; i < this->states.size(); ++i){
            particle state = this->states[i];
            int _x, _y, _width, _height;
            _x=MIN(MAX(cvRound(state.x),0),this->frame_size.width-this->reference_roi.width);
            _y=MIN(MAX(cvRound(state.y),0),this->frame_size.height-this->reference_roi.height);
            _width=MIN(MAX(cvRound(state.width),10),this->frame_size.width-_x);
            _height=MIN(MAX(cvRound(state.height),10),this->frame_size.height-_y);
            Rect current_state=Rect(_x, _y, _width, _height);
            samples.push_back(current_state);
    }
    vector<double> tmp_weights=this->detector.detect(current_frame,samples);
    double max_value = *max_element(tmp_weights.begin(), tmp_weights.end());
    for (size_t i = 0; i < this->states.size(); ++i){
            this->weights[i]+=log(tmp_weights[i]);
    }
    if(max_value/max_prob<0.8) this->detector.train(current_frame,update_roi);
    max_prob=max_value;
    this->resample();

}

double particle_filter::resample(){
    vector<double> cumulative_sum(n_particles);
    vector<double> normalized_weights(n_particles);
    vector<double> squared_normalized_weights(n_particles);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    double max_value = *max_element(this->weights.begin(), this->weights.end());
    double sumexp=0.0f;
    for (int i=0; i<n_particles; i++) {
        sumexp+=exp(weights.at(i)-max_value);
    }
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = exp(this->weights.at(i)-max_value-log(sumexp));
    }
    for (int i=0; i<n_particles; i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    //marginal_likelihood+=max_value+log(sum_weights[0])-log(n_particles); 
    ESS=1/sum_squared_weights[0]/n_particles;
    //cout  << "ESS :" << ESS << ",marginal_likelihood :" << marginal_likelihood <<  endl;
    //cout << "resampled particles!" << ESS << endl;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> new_states(n_particles);
        int i=0;
        while (i<n_particles) {
            double uni_rand = unif_rnd(generator);
            int ipos=0;
            while(cumulative_sum.at(ipos)<uni_rand) {
                //cout << ipos << ","<< cumulative_sum.at(ipos) << "," << uni_rand << endl;
                ipos++;
            }
            particle state=states.at(ipos);
            //cout << "pos:" << ipos << ", x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
            new_states[i]=state;
            weights[i]=1.0f/n_particles;
            i++;
        }
        states.swap(new_states);
        new_states = vector<particle>();
    }
    //else{
    //    weights.swap(log_normalized_weights);
    //}
    cumulative_sum.clear();
    normalized_weights.clear();
    squared_normalized_weights.clear();
    return marginal_likelihood;
}
