#include "bernoulli_particle_filter.hpp"

#ifndef PARAMS
const float POS_STD = 1.0;
const float OVERLAP_RATIO = 0.8;
const int NEWBORN_PARTICLES = 100;

const float SURVIVAL_PROB = 0.99;
const float INITIAL_EXISTENCE_PROB = 0.001;
const float BIRTH_PROB = 0.1;
#endif

BernoulliParticleFilter::~BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(){}

BernoulliParticleFilter::BernoulliParticleFilter(int n_particles){
	this->n_particles = n_particles;
	this->initialized = false;
	this->states.clear();
	//this->weights.clear();

	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);

	RowVectorXd theta_x_pos(2);
	theta_x_pos << POS_STD, POS_STD;
	this->theta_x.push_back(theta_x_pos);
	this->existence_prob = INITIAL_EXISTENCE_PROB;
}

bool BernoulliParticleFilter::is_initialized(){
	return this->initialized;
}

void BernoulliParticleFilter::initialize(Mat& current_frame, Rect ground_truth){
	this->img_size = current_frame.size();

	normal_distribution<double> random_x(0.0, this->theta_x.at(0)(0));
	normal_distribution<double> random_y(0.0, this->theta_x.at(0)(1));

	this->states.clear();
	//this->weights = VectorXd::Ones(this->n_particles + NEWBORN_PARTICLES) * log(1.0/(this->n_particles + NEWBORN_PARTICLES));
	this->weights = VectorXd::Ones(this->n_particles) * log(1.0/(this->n_particles));

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
		//float weight = log(1.0/(this->n_particles + NEWBORN_PARTICLES));
		for (int i = 0; i < this->n_particles; ++i)
		{
			particle state;
			float _x, _y, _width, _height;
			float _dx = random_x(this->generator);
			float _dy = random_y(this->generator);

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
			//this->weights.push_back(weight);
			this->ESS = 0.0f;
			this->sampleBox.push_back(Rect(state.x, state.y, state.width, state.height));
		}
	}

	Mat grayImg;
	cvtColor(current_frame, grayImg, CV_RGB2GRAY);

	this->haar.init(grayImg, this->reference_roi, this->sampleBox);

	this->initialized = true;
	cout << "initialized!!!" << endl;
}

void BernoulliParticleFilter::reinitialize(){
	this->initialized = false;
}

void BernoulliParticleFilter::predict(){
	/*VectorXd ps = VectorXd::Ones(this->states.size()) * log(SURVIVAL_PROB);
	VectorXd qs = VectorXd::Ones(this->states.size()) * log((1 - SURVIVAL_PROB));*/

	float new_existence_prob = this->existence_prob;

	//new_existence_prob = BIRTH_PROB * (1 - this->existence_prob) + ps.dot(this->weights) * this->existence_prob;
	new_existence_prob = BIRTH_PROB * (1 - this->existence_prob) + (log(SURVIVAL_PROB) * this->weights.sum() * this->existence_prob);

	/*********************** Generate birth particles ***********************/
	uniform_int_distribution<int> random_new_born_x(0, this->img_size.width);
	uniform_int_distribution<int> random_new_born_y(0, this->img_size.height);
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
		this->states.push_back(state);
		//this->weights.push_back(weight);
		this->sampleBox.push_back(Rect(state.x, state.y, state.width, state.height));
	}
	/************************************************************************/

}

void BernoulliParticleFilter::update(Mat& image){}

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


Rect BernoulliParticleFilter::estimate(Mat& image, bool draw){}