#include "smc_squared.hpp"

const float SHAPE=0.1;
const float SCALE=0.1;
const float HAAR_MU=1.0;
const float HAAR_SIG=1.0;
const float PRIOR_SD=0.01;
const float SMC_THRESHOLD=0.3;

smc_squared::smc_squared(int _n_particles,int _m_particles,int _fixed_lag,int _mcmc_steps){
    unsigned seed1= std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    n_particles=_n_particles;
    m_particles=_m_particles;
    fixed_lag=_fixed_lag;
    mcmc_steps=_mcmc_steps;
    initialized=false;
    theta_x_pos=MatrixXd::Zero(m_particles, 2);
    theta_x_scale=MatrixXd::Zero(m_particles, 2);
    
}

smc_squared::~smc_squared(){
    if(is_initialized()) filter_bank.clear();
    images = vector<Mat>();
}

void smc_squared::initialize(Mat& current_frame, Rect ground_truth){
    //cout << "initialize!" << endl;
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    particle_filter* filter=new particle_filter(n_particles);
    theta_weights.clear();
    filter->initialize(current_frame,ground_truth);
    theta_x=filter->get_dynamic_model();
    //cout << "smc_squared" << endl;
    float weight=1.0f;
    for(int j=0;j<m_particles;++j){
        //delete filter;
        particle_filter* new_filter=new particle_filter(n_particles);
        new_filter->initialize(current_frame,ground_truth);
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],SHAPE);
        prop_pos=prop_pos.array().abs().matrix();
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],SCALE);
        prop_std=prop_std.array().abs().matrix();
        theta_x_prop.push_back(prop_std);
        new_filter->update_model(theta_x_prop);
        //cout << "dynamic model proposal " << theta_x_prop[0].transpose() << ",scale model proposal " << theta_x_prop[1].transpose() << endl;
        theta_x_pos.row(j) = prop_pos;
        theta_x_scale.row(j) = prop_std;
        filter_bank.push_back(new_filter);
        theta_weights.push_back(weight);
    }
    images.clear();
    images.push_back(current_frame);
    reference_roi=ground_truth;
    initialized=true;
    estimates.clear();
    estimates.push_back(ground_truth);
}

bool smc_squared::is_initialized(){
    return initialized;
}

void smc_squared::reinitialize(){
    filter_bank.clear();
    initialized=false;
}

void smc_squared::predict(){
    for(int j=0;j<m_particles;++j){
        filter_bank[j]->predict();
    }
}

VectorXd smc_squared::proposal(VectorXd theta,double step_size){
    VectorXd proposal(theta.size());
    //double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<theta.size();i++){
        normal_distribution<double> random_walk(theta(i),step_size);
        //double val=MAX(random_walk(generator),eps);
        proposal[i] = random_walk(generator);
    }
    return proposal;
}

double smc_squared::igamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+=-b/x(i)-(a+1.0)*log(x(i))+a*log(b)-lgamma(a);
        }
    }
    return loglike;
}

double smc_squared::gamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+=-b/x(i)+(a-1.0)*log(x(i))-a*log(b)-lgamma(a);
        }
    }
    return loglike;
}


void smc_squared::update(Mat& current_frame){
    /*MVNGaussian theta_x_pos_proposal=MVNGaussian(theta_x_pos);
    MVNGaussian theta_x_scale_proposal=MVNGaussian(theta_x_scale);
    MVNGaussian theta_y_mu_proposal=MVNGaussian(theta_y_mu);
    MVNGaussian theta_y_sig_proposal=MVNGaussian(theta_y_sig);*/
    images.push_back(current_frame);
    vector<float> tmp_weights;
    for(int j=0;j<m_particles;++j){
        float weight=(float)theta_weights[j];
        filter_bank[j]->update(current_frame);
        tmp_weights.push_back(weight+filter_bank[j]->getMarginalLikelihood());
    }
    theta_weights.swap(tmp_weights);
    tmp_weights.clear();
    for(int j=0;j<m_particles;++j){
        //cout << " PMMH ---------------------" << endl;
        pmmh filter(n_particles,fixed_lag,mcmc_steps);
        theta_x=filter_bank[j]->get_dynamic_model();
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],SHAPE);
        prop_pos=prop_pos.array().abs().matrix();
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],SCALE);
        prop_std=prop_std.array().abs().matrix();
        theta_x_prop.push_back(prop_std);
        filter.initialize(images,estimates.front(),theta_x_prop);
        filter.run_mcmc();
        theta_x=filter.get_dynamic_model();
        filter_bank[j]->update_model(theta_x);
        //cout << "dynamic model proposal " << theta_x_prop[0].transpose() << ",scale model proposal " << theta_x_prop[1].transpose() << endl;
        //cout << "appearence mu proposal " << theta_y_prop[0].transpose() << endl;
        //cout << "appearence sig  proposal " << theta_y_prop[1].transpose() << endl;
    }
    resample();
}

Rect smc_squared::estimate(Mat& image,bool draw){
    float _x=0.0f,_y=0.0f,_width=0.0f,_height=0.0f;
    float norm=0.0f;
    for(int j=0;j<m_particles;++j){
        //float weight=(float)theta_weights[j];
        //float weight=1.0f/m_particles;
        Rect estimate=filter_bank[j]->estimate(image,draw);
        if(estimate.x>0 && estimate.x<images[0].cols 
            && estimate.y>0  && estimate.y<images[0].rows 
            && estimate.width>0 && estimate.width<images[0].cols 
            && estimate.height>0 && estimate.height<images[0].rows){
            _x+= estimate.x; 
            _y+= estimate.y; 
            _width+= estimate.width; 
            _height+= estimate.height;
            norm++;
        }
        //cout << j <<  ", weight:" << (float)theta_weights[j] << ",x:" << estimate.x << ",y:" << estimate.y << ",w:" << estimate.width << ",h:" << estimate.height << endl;
    }
    Point pt1,pt2;
    pt1.x=cvRound(_x/norm);
    pt1.y=cvRound(_y/norm);
    _width=cvRound(_width/norm);
    _height=cvRound(_height/norm);
    pt2.x=cvRound(pt1.x+_width);
    pt2.y=cvRound(pt1.y+_height);
    Rect new_estimate=Rect(pt1.x,pt1.y,_width,_height);
    //cout << "final estimate: " << new_estimate << endl;
    if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
    estimates.push_back(new_estimate);
    return new_estimate;
}

void smc_squared::draw_particles(Mat& image){
    for(int j=0;j<m_particles;++j){
        filter_bank[j]->draw_particles(image,Scalar(0,0,255));
    }
}

void smc_squared::resample(){
    vector<float> cumulative_sum(m_particles);
    vector<float> normalized_weights(m_particles);
    vector<float> new_weights(m_particles);
    vector<float> squared_normalized_weights(m_particles);
    uniform_real_distribution<float> unif_rnd(0.0,1.0); 
    float logsumexp=0.0f;
    float max_value = *max_element(theta_weights.begin(), theta_weights.end());
    for (unsigned int i=0; i<theta_weights.size(); i++) {
        new_weights[i]=exp(theta_weights[i]-max_value);
        logsumexp+=new_weights[i];
    }
    double norm_const=max_value+log(logsumexp);
    for (unsigned int i=0; i<theta_weights.size(); i++) {
        normalized_weights.at(i) = exp(theta_weights.at(i)-norm_const);
    }
    for (unsigned int i=0; i<theta_weights.size(); i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
        //cout << i << ", cumsum: " << normalized_weights.at(i) << "," <<cumulative_sum.at(i) << endl;
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    float ESS=(1.0f/sum_squared_weights[0])/m_particles;
    if(isless(ESS,(float)SMC_THRESHOLD)){
        vector<particle_filter*> new_filter_bank(m_particles);
        for (int i=0; i<m_particles; i++) {
            float uni_rand = unif_rnd(generator);
            vector<float>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            unsigned int ipos = distance(cumulative_sum.begin(), pos);
            //particle_filter* filter=filter_bank[ipos];
            new_filter_bank[i]=filter_bank[ipos];
            theta_weights.at(i)=1.0f;
        }
        filter_bank.swap(new_filter_bank);
    }
    else{
        theta_weights.swap(normalized_weights);
    }
    cumulative_sum.clear();
    normalized_weights.clear();
    new_weights.clear();
    squared_normalized_weights.clear();
}
