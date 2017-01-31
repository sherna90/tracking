#include "bernoulli_particle_filter.hpp"

#ifndef PARAMS
const float POS_STD = 1.0;
const float SCALE_STD = 1.0;
const float OVERLAP_RATIO = 0.8;
const float THRESHOLD = 1000;
const int NEWBORN_PARTICLES = 100;

const float SURVIVAL_PROB = 0.99;
const float INITIAL_EXISTENCE_PROB = 0.001;
const float BIRTH_PROB = 0.1;
const float DETECTION_RATE = 0.9;
const float CLUTTER_RATE = 1;
const float POSITION_LIKELIHOOD_STD = 10.0;
#endif

BernoulliParticleFilter::~BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(int n_particles){
	this->n_particles = n_particles;
	this->initialized = false;
	this->states.clear();
	this->weights.clear();

	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);

	RowVectorXd theta_x_pos(2);
	theta_x_pos << POS_STD, POS_STD;
	this->theta_x.push_back(theta_x_pos);

	RowVectorXd theta_x_scale(2);
	theta_x_scale << SCALE_STD,SCALE_STD;
	this->theta_x.push_back(theta_x_scale);

	this->existence_prob = INITIAL_EXISTENCE_PROB;
}

bool BernoulliParticleFilter::is_initialized(){
	return this->initialized;
}

void BernoulliParticleFilter::initialize(Mat& current_frame, Rect ground_truth){
	this->img_size = current_frame.size();

	normal_distribution<double> position_random_x(0.0, this->theta_x.at(0)(0));
	normal_distribution<double> position_random_y(0.0, this->theta_x.at(0)(1));

	this->states.clear();
	this->weights.clear();
	//this->weights = VectorXd::Ones(this->n_particles) * log(1.0/(this->n_particles));

	int left = MAX(ground_truth.x, 1);
	int top = MAX(ground_truth.y, 1);
	int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
	int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
	this->reference_roi = Rect(left, top, right - left, bottom - top);

	if ( (this->reference_roi.width > 0)
		&& ((this->reference_roi.x + this->reference_roi.width) < this->img_size.width)
		&& (this->reference_roi.height > 0)
		&& ((this->reference_roi.y + this->reference_roi.height) < this->img_size.height) )
	{
		double weight = -log(this->n_particles);
		for (int i = 0; i < this->n_particles; ++i)
		{
			particle state;
			float _x, _y, _width, _height;
			float _dx = position_random_x(this->generator);
			float _dy = position_random_y(this->generator);

			_x = MIN(MAX(cvRound(this->reference_roi.x + _dx), 0), this->img_size.width);
			_y = MIN(MAX(cvRound(this->reference_roi.y + _dy), 0), this->img_size.height);
			_width = MIN(MAX(cvRound(this->reference_roi.width), 10.0), this->img_size.width);
			_height = MIN(MAX(cvRound(this->reference_roi.height), 10.0), this->img_size.height);

			if ( ((_x + _width) < this->img_size.width)
				&& (_x > 0)
				&& ((_y + _height) < this->img_size.height)
				&& (_y > 0)
				&& (_width < this->img_size.width)
				&& (_height < this->img_size.height)
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
			//cout << "x: " << state.x << "\ty: " << state.y << "\twidth: " << state.width << "\theight: " << state.height << endl;
			this->states.push_back(state);
			this->weights.push_back(weight);
			this->ESS = 0.0f;
			//this->sampleBox.push_back(Rect(state.x, state.y, state.width, state.height));
		}
	}

	Mat grayImg;
	cvtColor(current_frame, grayImg, CV_RGB2GRAY);

	this->initialized = true;
	cout << "bernoulli particle filter initialized!!!" << endl;
}

void BernoulliParticleFilter::reinitialize(){
	this->initialized = false;
}

void BernoulliParticleFilter::predict(){
	/************** logsumexp **************/
	vector<double> normalized_weights(this->weights.size());
	double logsumexp = 0.0;
    double max_value = *max_element(this->weights.begin(), this->weights.end());
    for (size_t i = 0; i < this->weights.size(); i++) {
    	if (std::isnan(this->weights[i])) {
    		this->weights[i]=log(std::numeric_limits<double>::epsilon());
        }
        logsumexp += exp(this->weights[i] - max_value);
    }
    
    double norm_const = max_value + log(logsumexp);
    for (size_t i = 0; i < this->weights.size(); i++) {
        normalized_weights.at(i) = exp(this->weights.at(i) - norm_const);
    }
    /***************************************/
	//Scalar sum_weights = sum(this->weights);
	Scalar sum_weights = sum(normalized_weights);
    
	this->new_existence_prob = BIRTH_PROB * (1 - this->existence_prob) + (SURVIVAL_PROB * sum_weights[0] * this->existence_prob);

	cout << "old_existence_prob: " << this->existence_prob << endl;	
	cout << "new_existence_prob: " << this->new_existence_prob << endl;


	if (this->initialized)
	{
		vector<particle> tmp_states;
		vector<double> tmp_weights;
		/************************** Predict states **************************/
		normal_distribution<double> position_random_x(0.0, this->theta_x.at(0)(0));
		normal_distribution<double> position_random_y(0.0, this->theta_x.at(0)(1));
		normal_distribution<double> scale_random_width(0.0, this->theta_x.at(1)(0));
		uniform_real_distribution<double> unif(0.0,1.0);

		for (size_t i = 0; i < this->states.size(); ++i)
		{
			particle state = this->states[i];
			float _x, _y, _width, _height;
			
			do{
				float _dx = position_random_x(this->generator);
				float _dy = position_random_y(this->generator);

				_x = MIN(MAX(cvRound(state.x + _dx), 0), this->img_size.width);
				_y = MIN(MAX(cvRound(state.y + _dy), 0), this->img_size.height);
				_width = MIN(MAX(cvRound(state.width), 0), this->img_size.width);
				_height = MIN(MAX(cvRound(state.height), 0), this->img_size.height);
				
				state.x_p = state.x;
				state.y_p = state.y;
				state.width_p = state.width;
				state.height_p = state.height;
				state.scale_p = state.scale;
				
				state.x = _x;
				state.y = _y;
				state.width = _width;
				state.height = _height;
				state.scale = 2 * state.scale - state.scale_p + scale_random_width(this->generator);
			}
			while( ((_x + _width) < this->img_size.width)
				&& (_x > 0)
				&& ((_y + _height) < this->img_size.height)
				&& (_y > 0)
				&& (_width < this->img_size.width)
				&& (_height < this->img_size.height)
				&& (_width > 0)
				&& (_height > 0)
				&& (unif(this->generator) < SURVIVAL_PROB) );
			tmp_states.push_back(state);
			tmp_weights.push_back(log(this->existence_prob * SURVIVAL_PROB) + this->weights.at(i));
			//tmp_weights.push_back(this->weights.at(i));
		}
		/***********************************************************************/

		/*********************** Generate birth particles ***********************/
		uniform_int_distribution<int> random_new_born_x(0, this->img_size.width);
		uniform_int_distribution<int> random_new_born_y(0, this->img_size.height);
		//double weight = -log(NEWBORN_PARTICLES);
		for (int i = 0; i < NEWBORN_PARTICLES; ++i)
		{
			particle state;
			float _x, _y, _width, _height;
			do{
				_x = random_new_born_x(this->generator);
				_y = random_new_born_y(this->generator);
				_width = this->reference_roi.width;
				_height = this->reference_roi.height;
				
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
			while (!(((_x + _width) < this->img_size.width)
				&& (_x > 0)
				&& ((_y + _height) < this->img_size.height)
				&& (_y > 0)
				&& (_width < this->img_size.width)
				&& (_height < this->img_size.height)
				&& (_width > 0)
				&& (_height > 0)));
			//cout << "x: " << state.x << "\ty: " << state.y << "\twidth: " << state.width << "\theight: " << state.height << endl;
			tmp_states.push_back(state);
			tmp_weights.push_back(-log(NEWBORN_PARTICLES));
			//this->sampleBox.push_back(Rect(state.x, state.y, state.width, state.height));
		}
		this->weights.swap(tmp_weights);

		vector<double> normalized_weights(this->weights.size());
		double logsumexp = 0.0;
    	double max_value = *max_element(this->weights.begin(), this->weights.end());
    	for (size_t i = 0; i < this->weights.size(); i++) {
        	logsumexp += exp(this->weights[i] - max_value);
    	}
    	double norm_const = max_value + log(logsumexp);
    	for (size_t i = 0; i < this->weights.size(); i++) {
        	normalized_weights.at(i) = exp(this->weights.at(i) - norm_const);
    	}
		/************************************************************************/
		this->states.swap(tmp_states);
		this->weights.swap(normalized_weights);
	}

	/****************** limit range ******************/
	if(this->new_existence_prob > 0.999) this->new_existence_prob = 0.999;
	if(this->new_existence_prob < 0.001) this->new_existence_prob = 0.001;
	/*************************************************/

	/*cout << "predict" << endl;
	for (size_t i = 0; i < this->weights.size(); ++i)
    {
    	cout << this->weights.at(i) << ",";
    }*/

}

void BernoulliParticleFilter::update(Mat& image, vector<Rect> detections){
	if (detections.size() > 0)
	{
		vector<double> tmp_weights;
		MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);

		MatrixXd observations = MatrixXd::Zero(detections.size(), 4);
		for (size_t i = 0; i < detections.size(); i++){
            observations.row(i) << detections[i].x, detections[i].y, detections[i].width, detections[i].height;
        }

        MatrixXd psi(this->states.size(), detections.size());
        for (size_t i = 0; i < this->states.size(); ++i)
        {
        	particle state = this->states[i];
        	VectorXd mean(4);
        	mean << state.x, state.y, state.width, state.height;
        	MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
            MVNGaussian gaussian(mean, cov);
            double weight = this->weights[i];
            psi.row(i) = DETECTION_RATE * weight * gaussian.log_likelihood(observations).array().exp();
        }

        for (size_t i = 0; i < 100; ++i)
        {
        	cout << psi.row(i).sum() << ",";
        }
        cout << endl << "NEWBORN_PARTICLES" << endl;
        for (size_t i = 100; i < 200; ++i)
        {
        	cout << psi.row(i).sum() << ",";
        }

        VectorXd tau = VectorXd::Zero(detections.size());
        tau = psi.colwise().sum();
        
        for (size_t i = 0; i < detections.size(); ++i)
        {
            psi.col(i) = psi.col(i).array() / (tau(i) + CLUTTER_RATE);
        }

        VectorXd eta = psi.colwise().sum();

        for (size_t i = 0; i < this->weights.size(); ++i)
        {
            double weight = this->weights[i];
            //tmp_weights.push_back(log(weight * (1 - DETECTION_RATE) + eta.sum()));
            tmp_weights.push_back(weight * (1 - DETECTION_RATE) + eta.sum());
        }

        this->weights.swap(tmp_weights);

        /*for (size_t i = 0; i < this->weights.size(); ++i)
        {
        	cout << this->weights.at(i) << ",";
        }
        cout << endl;*/
        /************** update existence probability **************/
        Scalar sum_weights = sum(this->weights);

        double lambda_c = 20.0;
        double pdf_c = 1.6e-4;
        
        this->existence_prob =  (this->new_existence_prob * sum_weights[0]) / ( ( (lambda_c * pdf_c) * (1 - this->new_existence_prob) ) + ( this->new_existence_prob + sum_weights[0]) );
	    /****************** limit range ******************/
		if(this->existence_prob > 0.999) this->existence_prob = 0.999;
		if(this->existence_prob < 0.001) this->existence_prob = 0.001;
		/*************************************************/

        /**********************************************************/


        resample();
        tmp_weights.clear();

	}
}

void BernoulliParticleFilter::draw_particles(Mat& image, Scalar color){
    for (size_t i = 0; i < this->states.size(); i++){
        particle state = this->states[i];
        Point pt1, pt2;
        pt1.x = cvRound(state.x);
        pt1.y = cvRound(state.y);
        pt2.x = cvRound(state.x + state.width);
        pt2.y = cvRound(state.y + state.height);
        rectangle( image, pt1, pt2, color, 1, LINE_AA );
    }
}

void BernoulliParticleFilter::resample(){
	int num_states = this->states.size();
    //Scalar sum_weights = sum(this->weights);
    vector<double> cumulative_sum(num_states);
    vector<double> normalized_weights(num_states);
    vector<double> squared_normalized_weights(num_states);
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    
    Scalar sum_weights = sum(this->weights);

    for (size_t i = 0; i < this->weights.size(); i++) {
        //normalized_weights.at(i) = exp(this->weights.at(i) - norm_const);
        normalized_weights.at(i) = this->weights.at(i) / sum_weights[0];
    }
    
    for (size_t i = 0; i < this->weights.size(); i++) {
    	//cout <<  this->weights.at(i) << ", ";
        squared_normalized_weights.at(i) = normalized_weights.at(i) * normalized_weights.at(i);
        if (i == 0) {
            //cumulative_sum.at(i) = this->weights.at(i);
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            //cumulative_sum.at(i) = cumulative_sum.at(i - 1) + this->weights.at(i);
            cumulative_sum.at(i) = cumulative_sum.at(i - 1) + normalized_weights.at(i);
        }
    }
    
    Scalar sum_squared_weights = sum(normalized_weights);
    this->ESS = sum_squared_weights[0];
    cout << endl << "-------------------------" << endl;
    cout << "ESS: " << this->ESS << ", " << endl;
    vector<particle> new_states;
    vector<double> new_weights;
    for (int i = 0; i < this->n_particles; i++) {
        double uni_rand = unif_rnd(this->generator);
        vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
        int ipos = distance(cumulative_sum.begin(), pos);
        particle state = this->states[ipos];
        new_states.push_back(state);
        //this->weights.at(i) = double(sum_weights[0])/this->n_particles;
        new_weights.push_back(-log(this->n_particles));
    }

    this->states.swap(new_states);
    this->weights.swap(new_weights);

    cumulative_sum.clear();
    squared_normalized_weights.clear();
}


Rect BernoulliParticleFilter::estimate(Mat& image, bool draw){
	float _x = 0.0, _y = 0.0, _width = 0.0, _height = 0.0, norm = 0.0;
    Rect estimate;
    
    for (int i = 0;i < n_particles; i++){
        particle state = this->states[i];
        if( (state.x > 0)
        	&& (state.x < this->img_size.width)
            && (state.y > 0)
            && (state.y < this->img_size.height)
            && (state.width > 0)
            && (state.width < this->img_size.height)
            && (state.height > 0)
            && (state.height < this->img_size.height)){
            _x += state.x;
            _y += state.y;
            _width += state.width;
            _height += state.height;
            norm++;
        }
    }

    Point pt1,pt2;
    pt1.x = cvRound(_x/norm);
    pt1.y = cvRound(_y/norm);
    _width = cvRound(_width/norm);
    _height = cvRound(_height/norm);
    pt2.x = cvRound(pt1.x+_width);
    pt2.y = cvRound(pt1.y+_height); 
    if( (pt2.x < this->img_size.width)
    	&& (pt1.x >= 0)
    	&& (pt2.y < this->img_size.height)
    	&& (pt1.y >= 0)){
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
        estimate = Rect(pt1.x,pt1.y,_width,_height);
    }
    this->estimates.push_back(estimate);

    return estimate;
}