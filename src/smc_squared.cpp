#include "../include/smc_squared.hpp"

const float SHAPE=1.0;
const float SCALE=1.0;
const float PRIOR_SD=0.01;


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
    theta_y_mu=MatrixXd::Zero(m_particles, haar.featureNum);
    theta_y_sig=MatrixXd::Zero(m_particles, haar.featureNum);
    
}

smc_squared::~smc_squared(){
    if(is_initialized()) delete filter;
    images = vector<Mat>();
}

void smc_squared::initialize(Mat& current_frame, Rect ground_truth){
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    particle_filter filter=new particle_filter(n_particles);
    theta_weights.clear();
    filter->initialize(current_frame,ground_truth);
    theta_x=filter->get_dynamic_model();
    theta_y=filter->get_observation_model();
    haar<-filter.haar;
    double weight=1.0/m_particles;
    for(int j=0;j<m_particles;++j){
        delete filter;
        filter=new particle_filter(n_particles);
        filter->initialize(current_frame,ground_truth,haar);
        filter.haar->haar;
        theta_y_prop.clear();
        VectorXd prop_mu=proposal(theta_y[0],100.0);
        theta_y_prop.push_back(prop_mu);
        VectorXd prop_sig=proposal(theta_y[1],10.0);
        prop_sig=prop_sig.array().abs().matrix();
        theta_y_prop.push_back(prop_sig);
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],1.0);
        prop_pos=prop_pos.array().abs().matrix();
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],0.01);
        prop_std=prop_std.array().abs().matrix();
        theta_x_prop.push_back(prop_std);
        filter_bank.push_back(filter);
        theta_x_pos.row(j) << prop_pos;
        theta_y_pos.row(j) << prop_std;
        theta_y_mu.row(j) << prop_mu;
        theta_y_sig.row(j) << prop_sig;
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
    delete filter;
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
    images.push_back(current_frame);
    vector<double> tmp_weights;
    for(int j=0;j<m_particles;++j){
        filter_bank[j]->update(current_frame);
        tmp_weights.push_back(filter_bank[j].getMarginalLikelihood());
    }
    theta_weights.swap(tmp_weights);
    tmp_weights.clear();
    resample();
}

Rect smc_squared::estimate(Mat& image,bool draw){
    int _x=0,_y=0,_width=0,_height=0;
    for(int j=0;j<m_particles;++j){
        Rect estimate filter_bank[j]->estimate(image,draw);
        _x+=theta_wight[j]*estimate.x;
        _y+=theta_wight[j]*estimate.y;
        _width+=theta_wight[j]*estimate.width;
        _height+=theta_wight[j]*estimate.height;
    }
    Rect estimate(cvRound(_x), cvRound(_y), cvRound(_width), cvRond(_height));
    estimates.push_back(estimate);
    return estimate;
}

void smc_squared::draw_particles(Mat& image){
     //filter->draw_particles(image,Scalar(0,255,255));
}

void smc_squared::resample(){
    vector<double> cumulative_sum(m_particles);
    vector<double> normalized_weights(m_particles);
    vector<double> new_weights(m_particles);
    vector<double> squared_normalized_weights(m_particles);
    uniform_real_distribution<double> unif_rnd(0.0,1.0); 
    double logsumexp=0.0;
    double max_value = *max_element(theta_weights.begin(), theta_weights.end());
    for (unsigned int i=0; i<theta_weights.size(); i++) {
        new_weights[i]=exp(theta_weights[i]-max_value);
        logsumexp+=new_weights[i];
    }
    double norm_const=max_value+log(logsumexp);
    for (unsigned int i=0; i<weights.size(); i++) {
        normalized_weights.at(i) = exp(theta_weights.at(i)-norm_const);
    }
    for (unsigned int i=0; i<theta_weights.size(); i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    marginal_likelihood+=norm_const-log(m_particles); 
    ESS=(1.0f/sum_squared_weights[0])/m_particles;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle_filter*> new_filter_bank;
        for (int i=0; i<n_particles; i++) {
            double uni_rand = unif_rnd(generator);
            vector<double>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            int ipos = distance(cumulative_sum.begin(), pos);
            particle_filter filter=filter_bank[ipos];
            new_filter_bank.push_back(filter);
            weights.at(i)=log(1.0f/m_particles);
        }
        filter_bank.swap(new_filter_bank);
    }
    else{
        theta_weights.swap(new_weights);
    }
    cumulative_sum.clear();
    normalized_weights.clear();
    new_weights.clear();
    squared_normalized_weights.clear();
}
