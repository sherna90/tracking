#include "bernoulli_particle_filter_HMC.hpp"

#ifndef PARAMS
const float POS_STD = 2.0;
const float SCALE_STD = 0.1;
const float OVERLAP_RATIO = 0.8;
const float THRESHOLD = 1000;
const int NEWBORN_PARTICLES = 0;

const float SURVIVAL_PROB = 0.99;
const float INITIAL_EXISTENCE_PROB = 0.99;
const float BIRTH_PROB = 0.1;
const float DETECTION_RATE = 0.9;
const float CLUTTER_RATE = 1;
const float POSITION_LIKELIHOOD_STD = 10.0;
const float LAMBDA_C= 20.0;
const float PDF_C = 1.6e-4;

const double LAMBDA_BC=20.4;

const double GROUP_THRESHOLD = 0.1;
const double HIT_THRESHOLD = 0.2;

//const int this->step_slide = 20;
#endif

BernoulliParticleFilter::~BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(int n_particles, double lambda, double mu, double epsilon){
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

void BernoulliParticleFilter::initialize(const Mat& current_frame, const Rect ground_truth){
	vector<Rect> sample_boxes;
	this->img_size = current_frame.size();

	normal_distribution<double> position_random_x(0.0, this->theta_x.at(0)(0));
	normal_distribution<double> position_random_y(0.0, this->theta_x.at(0)(1));

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
	//sample_boxes.push_back(this->reference_roi);
	if ( (this->reference_roi.width > 0)
		&& ((this->reference_roi.x + this->reference_roi.width) < this->img_size.width)
		&& (this->reference_roi.height > 0)
		&& ((this->reference_roi.y + this->reference_roi.height) < this->img_size.height) )
	{
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
			//cout << "x: " << state.x << "\ty: " << state.y << "\twidth: " << state.width << "\theight: " << state.height << "\tweight: " << weight << endl;
			this->states.push_back(state);
			this->weights.push_back(weight);
			sample_boxes.push_back(Rect(state.x,state.y,state.width,state.height));
		}

	Mat current_frame_copy;
	current_frame.copyTo(current_frame_copy);
	//Mat current_frame_copy;
    //cvtColor(current_frame, current_frame_copy, CV_RGB2GRAY);
    this->detector.init(GROUP_THRESHOLD, HIT_THRESHOLD, this->reference_roi);
    this->detector.train(current_frame_copy, this->reference_roi);
    this->initialized = true;
    cout << "initialized!!!" << endl;
	}
}

void BernoulliParticleFilter::reinitialize(){
	this->initialized = false;
}

void BernoulliParticleFilter::predict(){
	/************** logsumexp **************/

	Scalar sum_weights = sum(this->weights);
	this->new_existence_prob = BIRTH_PROB * (1 - this->existence_prob) + (SURVIVAL_PROB * sum_weights[0] * this->existence_prob);
	if (this->initialized)
	{
		vector<particle> tmp_states;
		vector<double> tmp_weights;
		/************************** Predict states **************************/
		normal_distribution<double> position_random_x(0.0, this->theta_x.at(0)(0));
		normal_distribution<double> position_random_y(0.0, this->theta_x.at(0)(1));
		normal_distribution<double> scale_random_width(0.0, this->theta_x.at(1)(0));
		normal_distribution<double> scale_random_height(0.0, this->theta_x.at(1)(1));
		uniform_real_distribution<double> unif(0.0,1.0);

		for (size_t i = 0; i < this->states.size(); ++i)
		{
			particle state = this->states[i];
			float _x, _y, _width, _height;
			float _dx = position_random_x(this->generator);
			float _dy = position_random_y(this->generator);
			float _dw = scale_random_width(this->generator);
			float _dh = scale_random_height(this->generator);
			_x = MIN(MAX(cvRound(state.x + _dx), 0), this->img_size.width);
			_y = MIN(MAX(cvRound(state.y + _dy), 0), this->img_size.height);
			_width = MIN(MAX(cvRound(state.width+_dw), 0), this->img_size.width);
			_height = MIN(MAX(cvRound(state.height+_dh), 0), this->img_size.height);
			if(((_x + _width) < this->img_size.width)
				&& (_x > 0)
				&& ((_y + _height) < this->img_size.height)
				&& (_y > 0)
				&& (_width < this->img_size.width)
				&& (_height < this->img_size.height)
				&& (_width > 0)
				&& (_height > 0)){
				state.x_p = state.x;
				state.y_p = state.y;
				state.width_p = state.width;
				state.height_p = state.height;
				state.scale_p = state.scale;
				state.x = _x;
				state.y = _y;
				state.width = _width;
				state.height = _height;
				state.scale = 2 * state.scale - state.scale_p;
			}
			else{
				state.x = state.x;
                state.y = state.y;
                state.x_p = this->reference_roi.x;
                state.y_p = this->reference_roi.y;
                state.width = cvRound(this->reference_roi.width);
                state.height = cvRound(this->reference_roi.height);
                state.width_p = cvRound(this->reference_roi.width);
                state.height_p = cvRound(this->reference_roi.height);
                state.scale = 1.0f;
			}
			tmp_states.push_back(state);
			tmp_weights.push_back(SURVIVAL_PROB*this->existence_prob*this->weights.at(i));
			//tmp_weights.push_back(this->weights.at(i));
		}
		/***********************************************************************/

		/*********************** Generate birth particles ***********************/
		uniform_int_distribution<int> random_new_born_x(0, this->img_size.width - this->reference_roi.width);
		uniform_int_distribution<int> random_new_born_y(0, this->img_size.height - this->reference_roi.width);
		double nb_weight = BIRTH_PROB * (1 - this->existence_prob)/NEWBORN_PARTICLES;
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
			//cout << "x: " << state.x << "\ty: " << state.y << "\twidth: " << state.width << "\theight: " << state.height << "\tweight: " << nb_weight << endl;
			this->states.push_back(state);
			tmp_states.push_back(state);
			tmp_weights.push_back(nb_weight);
		}
		this->weights.swap(tmp_weights);
		vector<double> normalized_weights(this->weights.size());
		Scalar sum_weights = sum(this->weights);
    	for (size_t i = 0; i < this->weights.size(); i++) {
        	normalized_weights.at(i) = this->weights.at(i)/sum_weights[0];
    	}
		/************************************************************************/
		this->states.swap(tmp_states);
		//this->weights.swap(normalized_weights);
	}

	/****************** limit range ******************/
	if(this->new_existence_prob > 0.999) this->new_existence_prob = 0.999;
	if(this->new_existence_prob < 0.001) this->new_existence_prob = 0.001;
	/*************************************************/

	//exit(EXIT_FAILURE);
	//cout << "predicted!!!" << endl;
}

void BernoulliParticleFilter::update(const Mat& image){
	Mat current_frame;
	image.copyTo(current_frame);
	//Mat current_frame_copy;
	//cvtColor(image, current_frame_copy, CV_RGB2GRAY);

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
	this->dppResults = this->detector.detect(current_frame,update_roi);
	//this->detector.train(current_frame,update_roi);
	/*//MatrixXd featureValues = this->detector.getFeatureValues();
	//VectorXd phi = this->detector.getDetectionWeights();
	//cout << phi.rows() << "," << featureValues.rows() << endl;
	//VectorXd penalty_weights=VectorXd::Zero(this->preDetections.size());
	//for(unsigned int i = 0; i<this->preDetections.size();i++){
	//	Rect intersection = update_roi & this->preDetections[i];
	//	penalty_weights(i) = (double)intersection.area()/update_roi.area();
	//}
   	//VectorXd qualityTerm;
   	//this->dppResults = this->dpp.run(this->preDetections, phi,penalty_weights,featureValues, qualityTerm, this->lambda, this->mu, this->epsilon);
	cout << this->dppResults.size()  << endl;*/
	

	if (this->dppResults.size() > 0)
	{
		vector<double> tmp_weights;
		MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);

		MatrixXd observations = MatrixXd::Zero(this->dppResults.size(), 4);
		for (size_t i = 0; i < this->dppResults.size(); i++){
            observations.row(i) << this->dppResults[i].x, this->dppResults[i].y, this->dppResults[i].width, this->dppResults[i].height;
            rectangle( image, Point(this->dppResults[i].x, this->dppResults[i].y), Point(this->dppResults[i].x+this->dppResults[i].width, this->dppResults[i].y+this->dppResults[i].height), Scalar(0,255,255), 1, LINE_AA );
      
        }

        MatrixXd psi(this->states.size(), this->dppResults.size());

        for (size_t i = 0; i < this->states.size(); ++i)
        {
        	particle state = this->states[i];
        	VectorXd mean(4);
        	mean << state.x, state.y, state.width, state.height;
        	MatrixXd cov = POSITION_LIKELIHOOD_STD * POSITION_LIKELIHOOD_STD * MatrixXd::Identity(4, 4);
            MVNGaussian gaussian(mean, cov);
            //double weight = this->weights[i];

            psi.row(i) = gaussian.log_likelihood(observations).array().exp();
        }


        VectorXd tau = VectorXd::Zero(this->dppResults.size());
        tau = psi.colwise().sum();
        VectorXd eta = psi.colwise().sum();

        for (size_t i = 0; i < this->weights.size(); ++i)
        {
            double weight = this->weights[i];
            tmp_weights.push_back(weight * (1 - DETECTION_RATE) + psi.row(i).sum()/( LAMBDA_C * PDF_C));
        }
		this->weights.swap(tmp_weights);
    	Scalar sum_weights = sum(this->weights);
        this->existence_prob =  (this->new_existence_prob * sum_weights[0]) / ( ( (LAMBDA_C * PDF_C) * (1 - this->new_existence_prob) ) + ( this->new_existence_prob + sum_weights[0]) );
	    if(this->existence_prob > 0.999) this->existence_prob = 0.999;
		if(this->existence_prob < 0.001) this->existence_prob = 0.001;
        resample();
        tmp_weights.clear();
	}
	//this->reference_hist=float(1-1./8.)*this->reference_hist.array()+float(1./8.)*this->featureValues.row(0).array();
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
	uniform_real_distribution<double> unif_rnd(0.0,1.0);
	int num_states = this->states.size();
	vector<double> normalized_weights(this->weights.size());
	Scalar sum_weights = sum(this->weights);
    for (size_t i = 0; i < this->weights.size(); i++) {
        normalized_weights.at(i) =this->weights.at(i)/sum_weights[0];
    }
    vector<double> cumulative_sum(num_states);
    for (size_t i = 0; i < this->weights.size(); i++) {
        if (i == 0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i - 1) + normalized_weights.at(i);
        }
    }
    vector<particle> new_states;
    vector<double> new_weights;
    for (int i = 0; i < this->n_particles; i++) {
        double uni_rand = unif_rnd(this->generator);
        vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
        int ipos = distance(cumulative_sum.begin(), pos);
        particle state = this->states[ipos];
        new_states.push_back(state);
        //this->weights.at(i) = double(sum_weights[0])/this->n_particles;
        new_weights.push_back(1.0f/this->n_particles);
    }

    this->states.swap(new_states);
    this->weights.swap(new_weights);

    cumulative_sum.clear();
    normalized_weights.clear();
    /*cout << "\nresample" << endl;
	for (size_t i = 0; i < this->weights.size(); ++i)
    {
    	cout << this->weights.at(i) << ",";
    }*/
    //exit(EXIT_FAILURE);
}


Rect BernoulliParticleFilter::estimate(const Mat& image, bool draw){
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

    this->reference_roi = estimate;

    this->estimates.push_back(estimate);

    return estimate;
}