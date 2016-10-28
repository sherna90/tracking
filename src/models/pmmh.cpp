#include "pmmh.hpp"

const float SHAPE=0.1;
const float SCALE=0.1;
const float HAAR_MU=1.0;
const float HAAR_SIG=1.0;
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

void pmmh::initialize(vector<Mat> _images, Rect ground_truth){
    //std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(n_particles);
    images=_images;
    filter->initialize(images[0],ground_truth);
    reference_roi=ground_truth;
    theta_x=filter->get_dynamic_model();
    initialized=true;
    estimates.clear();
    estimates.push_back(ground_truth);
    matrix_pos=MatrixXd::Zero(mcmc_steps, 2);
    matrix_width=MatrixXd::Zero(mcmc_steps, 2);
    matrix_haar_mu=MatrixXd::Zero(mcmc_steps, filter->haar.featureNum);
    matrix_haar_std=MatrixXd::Zero(mcmc_steps, filter->haar.featureNum);
}

void pmmh::initialize(vector<Mat> _images, Rect ground_truth,vector<VectorXd> _theta_x){
    //std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(n_particles);
    images=_images;
    filter->initialize(images[0],ground_truth);
    reference_roi=ground_truth;
    theta_x=_theta_x;
    filter->update_model(theta_x);
    initialized=true;
    estimates.clear();
    estimates.push_back(ground_truth);
    matrix_pos=MatrixXd::Zero(mcmc_steps, 2);
    matrix_width=MatrixXd::Zero(mcmc_steps, 2);
}
bool pmmh::is_initialized(){
    return initialized;
}

void pmmh::reinitialize(Mat& current_frame, Rect ground_truth){
    if(mcmc_steps>0){
        theta_x=get_dynamic_model();
    }
    if(is_initialized()) delete filter;
    filter=new particle_filter(n_particles);
    filter->initialize(current_frame,ground_truth);
    filter->update_model(theta_x);
}

void pmmh::predict(){
    filter->predict();
}

void pmmh::update(Mat& image){
    filter->update(image);
}

double pmmh::marginal_likelihood(vector<VectorXd> theta_x){
    particle_filter proposal_filter(n_particles);
    //int data_size=(int)images.size();
    //int data_size=fixed_lag;
    int data_size=(fixed_lag >= (int)images.size() || fixed_lag==0) ? (int)images.size() : fixed_lag;
    int time_step= 0 ;
    Mat current_frame = images.front().clone(); 
    proposal_filter.initialize(current_frame,estimates.front());
    proposal_filter.update_model(theta_x);
    for(int k=time_step;k<data_size;++k){
        //cout << "time step:" << k << ", ML: " << proposal_filter.getMarginalLikelihood() << endl;
        current_frame = images.at(k).clone();
        proposal_filter.predict();
        proposal_filter.update(current_frame);
    }
    double res=proposal_filter.getMarginalLikelihood();
    return res;
}

VectorXd pmmh::proposal(VectorXd theta,double step_size){
    VectorXd proposal(theta.size());
    proposal.setZero();
    double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<theta.size();i++){
        normal_distribution<double> random_walk(theta(i),step_size);
        double val=MAX(random_walk(generator),eps);
        proposal[i] = val;
    }
    return proposal;
}

double pmmh::igamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            //cout << loglike << ", " << x(i) << "," << a << "," << b << endl;
            x(i)+=eps;
            loglike+= -b / x(i)  - (a + 1.0) * log(x(i)) + a * log(b) - lgamma(a);
        }
    }
    return loglike;
}

double pmmh::gamma_prior(VectorXd x, double a, double b)
{
    double loglike=0.0;
    for(int i=0;i<x.size();i++){
        if (x(i) >= 0 && a >= 0 && b >= 0){
            loglike+= -x(i) * b + (a - 1.0) * log(x(i)) + a * log(b) - lgamma(a);
        }
    }
    return loglike;
}


void pmmh::run_mcmc(){
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    double forward_filter = marginal_likelihood(theta_x);
    double accept_rate=0;
    for(int n=0;n<mcmc_steps;n++){
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],SHAPE);
        prop_pos=prop_pos.array().abs().matrix();
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],SCALE);
        prop_std=prop_std.array().abs().matrix();
        theta_x_prop.push_back(prop_std);
        double proposal_filter = marginal_likelihood(theta_x_prop);
        double acceptprob = proposal_filter - forward_filter;
        acceptprob+=igamma_prior(prop_pos,SHAPE,SCALE)-igamma_prior(theta_x.at(0),SHAPE,SCALE);
        acceptprob+=igamma_prior(prop_std,SHAPE,SCALE)-igamma_prior(theta_x.at(1),SHAPE,SCALE);
        //if(n % 100 == 0){
            //cout <<"Iter: "<< n << ", accept rate: " <<  accept_rate/n <<  endl;       
        //}
        double u=unif_rnd(generator);
        if( isfinite(proposal_filter) 
            &&  isfinite(forward_filter) 
            && (log(u) < acceptprob) 
            && (theta_x_prop.at(0).array()>0).all() 
            && (theta_x_prop.at(1).array()>0).all() 
            ){
            theta_x=theta_x_prop;
            filter->update_model(theta_x);
            forward_filter=proposal_filter;
            accept_rate++;
            }
        else {
            theta_x_prop=theta_x;
        }
        matrix_pos.row(n)=theta_x_prop.at(0).transpose() ;
        matrix_width.row(n)=theta_x_prop.at(1).transpose();  
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

vector<VectorXd> pmmh::get_dynamic_model(){
    theta_x_prop.clear();
    VectorXd prop_pos=matrix_pos.colwise().sum()/mcmc_steps;
    theta_x_prop.push_back(prop_pos);
    VectorXd prop_std=matrix_width.colwise().sum()/mcmc_steps;
    theta_x_prop.push_back(prop_std);
    return theta_x_prop;
}

