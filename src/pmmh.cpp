#include "../include/pmmh.hpp"

const float SHAPE=1.0;
const float SCALE=1.0;
const float PRIOR_SD=0.01;


pmmh::pmmh(int _n_particles,int _fixed_lag,int _mcmc_steps){
    unsigned seed1= std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    n_particles=_n_particles;
    fixed_lag=_fixed_lag;
    mcmc_steps=_mcmc_steps;
    initialized=false;
}

pmmh::~pmmh(){
    if(is_initialized()) delete filter;
    images = vector<Mat>();
}

void pmmh::initialize(Mat& current_frame, Rect ground_truth){
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(n_particles);
    filter->initialize(current_frame,ground_truth);
    images.clear();
    images.push_back(current_frame);
    reference_roi=ground_truth;
    theta_x=filter->get_dynamic_model();
    theta_y=filter->get_observation_model();
    initialized=true;
    estimates.clear();
    estimates.push_back(ground_truth);
}

bool pmmh::is_initialized(){
    return initialized;
}

void pmmh::reinitialize(){
    delete filter;
    initialized=false;
}

void pmmh::predict(){
    filter->predict();
}

double pmmh::marginal_likelihood(vector<VectorXd> theta_x,vector<VectorXd> theta_y){
    particle_filter pmmh(n_particles);
    int data_size=(int)images.size();
    //int time_step=(fixed_lag>=data_size)? 0 : data_size-fixed_lag;
    //cout << time_step << ","<< estimates.size() << "," << data_size << endl;
    int time_step=0;
    Mat current_frame = images.at(time_step).clone(); 
    pmmh.initialize(current_frame,estimates.at(time_step));
    pmmh.haar=filter->haar;
    pmmh.update_model(theta_x,theta_y);
    for(int k=time_step+1;k<data_size;++k){
        current_frame = images.at(k).clone();
        pmmh.predict();
        pmmh.update(current_frame);
    }
    //cout << "ML:" << pmmh.getMarginalLikelihood();
    //cout << endl;
    double res=pmmh.getMarginalLikelihood();
    return res;
}

VectorXd pmmh::proposal(VectorXd theta,double step_size){
    VectorXd proposal(theta.size());
    //double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<theta.size();i++){
        normal_distribution<double> random_walk(theta(i),step_size);
        //double val=MAX(random_walk(generator),eps);
        proposal[i] = random_walk(generator);
    }
    return proposal;
}

double pmmh::igamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+=-b/x(i)-(a+1.0)*log(x(i))+a*log(b)-lgamma(a);
        }
    }
    return loglike;
}

double pmmh::gamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+=-b/x(i)+(a-1.0)*log(x(i))-a*log(b)-lgamma(a);
        }
    }
    return loglike;
}


void pmmh::update(Mat& current_frame){
    images.push_back(current_frame);
    filter->update(current_frame); 
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    //double forward_filter = filter->getMarginalLikelihood();
    //cout << "----------------" << endl;
    double forward_filter = marginal_likelihood(theta_x,theta_y);
    for(int n=0;n<mcmc_steps;n++){
        theta_y_prop.clear();
        VectorXd prop_mu=proposal(theta_y[0],100.0);
        theta_y_prop.push_back(prop_mu);
        VectorXd prop_sig=proposal(theta_y[1],10.0);
        theta_y_prop.push_back(prop_sig);
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],0.1);
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],0.01);
        theta_x_prop.push_back(prop_std);
        double proposal_filter = marginal_likelihood(theta_x_prop,theta_y_prop);
        double acceptprob = -proposal_filter + forward_filter;
        acceptprob+=-gamma_prior(prop_sig,SHAPE,SCALE)+gamma_prior(theta_y.at(1),SHAPE,SCALE);
        acceptprob+=-gamma_prior(prop_pos,SHAPE,SCALE)+gamma_prior(theta_x.at(0),SHAPE,SCALE);
        acceptprob+=-gamma_prior(prop_std,SHAPE,SCALE)+gamma_prior(theta_x.at(1),SHAPE,SCALE);
        double u=unif_rnd(generator);
        //cout << forward_filter << "," << proposal_filter << "," << acceptprob << "," << log(u) << endl;
        if( isfinite(proposal_filter) 
            &&  isfinite(forward_filter) 
            && log(u) < acceptprob 
            && (theta_x_prop.at(0).array()>0).all() 
            && (theta_x_prop.at(1).array()>0).all() 
            && (theta_y_prop.at(1).array()>0).all()){
            theta_y=theta_y_prop;
            theta_x=theta_x_prop;
            filter->update_model(theta_x,theta_y);
            forward_filter=proposal_filter;
            //cout  <<theta_y_prop.at(0).transpose()  <<  endl;
            //cout  <<theta_x_prop.at(0).transpose()  << "," <<theta_x_prop.at(1).transpose() << endl;
            //cout << "theta_y_mu:" << theta_y_prop.at(0).transpose() << endl;
            //cout << "theta_y_sig:" << theta_y_prop.at(1).transpose() << endl;
            }
    }
   
}

Rect pmmh::estimate(Mat& image,bool draw){
    Rect estimate=filter->estimate(image,draw);
    estimates.push_back(estimate);
    return estimate;
}

void pmmh::draw_particles(Mat& image){
     filter->draw_particles(image,Scalar(0,255,255));
}
