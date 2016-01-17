#include "../include/pmmh.hpp"

const float SHAPE=0.1;
const float SCALE=1.0;
const float PRIOR_SD=0.1;


pmmh::pmmh(int _num_particles,int _fixed_lag,int _mcmc_steps){
    unsigned seed1= std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    num_particles=_num_particles;
    fixed_lag=_fixed_lag;
    mcmc_steps=_mcmc_steps;
    initialized=false;
}

pmmh::~pmmh(){
    if(is_initialized()) delete filter;
    images = vector<Mat>();
}

void pmmh::initialize(Mat& current_frame, Rect ground_truth){
    images = vector<Mat>();
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(num_particles);
    filter->initialize(current_frame,ground_truth);
    images.push_back(current_frame);
    reference_roi=ground_truth;
    theta_y.resize(2);
    theta_y(0)=prior(generator);
    theta_y(1)=prior(generator);
    theta_x.resize(2);
    theta_x(0)=prior(generator);
    theta_x(1)=prior(generator);
    //filter->update_model(theta_x,theta_y);
    theta_x=filter->get_dynamic_model();
    theta_y=filter->get_observation_model();
    initialized=true;
}

bool pmmh::is_initialized(){
    return initialized;
}

void pmmh::reinitialize(){
    delete filter;
    initialized=false;
}

double pmmh::marginal_likelihood(VectorXd theta_x,VectorXd theta_y){
    particle_filter pmmh(num_particles);
    Mat current_frame = images[0].clone(); 
    pmmh.initialize(current_frame,reference_roi);
    pmmh.update_model(theta_x,theta_y);
    int time_step=MIN((int)images.size(),fixed_lag);
    //int time_step=filter->time_stamp;   
    double res=0.0;
    for(int k=0;k<time_step;k++){
        cout << "time:" << k << endl;
        Mat current_frame = images[k].clone();
        pmmh.predict();
        pmmh.update(current_frame);
    }
    res=pmmh.getMarginalLikelihood();
    return res;
}

VectorXd pmmh::continuous_proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<alpha.size();i++){
        normal_distribution<double> random_walk(alpha(i),PRIOR_SD);
        double val=MAX(random_walk(generator),eps);
        proposal[i] = val;
    }
    return proposal;
}

double pmmh::gamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+=-b/x(i)-(a+1.0)*log(x(i))+a*log(b)-lgamma(a);
        }
    }
    return loglike;
}

void pmmh::update(Mat& current_frame){
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    filter->predict();
    filter->update(current_frame);
    //double forward_filter = filter->getMarginalLikelihood();
    double forward_filter = marginal_likelihood(theta_x,theta_y);
        
    images.push_back(current_frame);
    for(int n=0;n<mcmc_steps;n++){
        theta_y_prop=continuous_proposal(theta_y);
        theta_x_prop=continuous_proposal(theta_x);
        double proposal_filter = marginal_likelihood(theta_x_prop,theta_y_prop);
        double acceptprob = proposal_filter - forward_filter;
        acceptprob+=gamma_prior(theta_y_prop,SHAPE,SCALE)-gamma_prior(theta_y,SHAPE,SCALE);
        cout << "--------------------------"  << endl;
        cout << "MCMC Step:" << n << endl;
        cout << "filter ll:" << forward_filter << endl;
        cout << "proposal ll:" << proposal_filter << endl;
        cout << "theta_x:" << theta_x << ", prop:" << theta_x_prop << endl;
        cout << "theta_y:" << theta_y << ", prop:" << theta_y_prop << endl;    
        double u=unif_rnd(generator);
        if( isfinite(proposal_filter) &&  isfinite(forward_filter) 
            && log(u) < acceptprob && (theta_y_prop.array()>0).all() 
            && (theta_x_prop.array()>0).all()){
            cout << "accept!" << endl;
            theta_y=theta_y_prop;
            theta_x=theta_x_prop;
            filter->update_model(theta_x_prop,theta_y_prop);
            forward_filter=proposal_filter;
            }
    }
    
}

Rect pmmh::estimate(Mat& image,bool draw){
    return filter->estimate(image,draw);
}

void pmmh::draw_particles(Mat& image){
     filter->draw_particles(image);
}