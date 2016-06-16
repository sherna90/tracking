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

void pmmh::initialize(vector<Mat> _images, Rect ground_truth){
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(n_particles);
    images=_images;
    filter->initialize(images[0],ground_truth);
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

void pmmh::reinitialize(Mat& current_frame, Rect ground_truth){
    MatrixXd matrix_pos,matrix_width,matrix_haar_mu,matrix_haar_std,matrix_color;
    string file1("matrix_pos.txt");
    string file2("matrix_width.txt");
    string file3("matrix_haar_mu.txt");
    string file4("matrix_haar_std.txt");
    string file5("matrix_color.txt");
    read_data(file1,matrix_pos,mcmc_steps, 2);
    read_data(file2,matrix_width,mcmc_steps, 2);
    read_data(file3,matrix_haar_mu,mcmc_steps, filter->haar.featureNum);
    read_data(file4,matrix_haar_std,mcmc_steps, filter->haar.featureNum);
    read_data(file5,matrix_color,mcmc_steps, H_BINS*S_BINS);
    theta_y_prop.clear();
    VectorXd prop_mu=matrix_haar_mu.colwise().sum()/mcmc_steps;
    theta_y_prop.push_back(prop_mu);
    VectorXd prop_sig=matrix_haar_std.colwise().sum()/mcmc_steps;
    theta_y_prop.push_back(prop_sig);
    VectorXd prop_color=matrix_color.colwise().sum()/mcmc_steps;
    theta_y_prop.push_back(prop_color/prop_color.sum());
    theta_x_prop.clear();
    VectorXd prop_pos=matrix_pos.colwise().sum()/mcmc_steps;
    theta_x_prop.push_back(prop_pos);
    VectorXd prop_std=matrix_width.colwise().sum()/mcmc_steps;
    theta_x_prop.push_back(prop_std);
    filter->initialize(current_frame,ground_truth);
    filter->update_model(theta_x_prop,theta_y_prop);
}

void pmmh::predict(){
    filter->predict();
}

void pmmh::update(Mat& image){
    filter->update(image);
}

double pmmh::marginal_likelihood(vector<VectorXd> theta_x,vector<VectorXd> theta_y){
    particle_filter proposal_filter(n_particles);
    int data_size=(int)images.size();
    //int time_step=(fixed_lag>=data_size)? 0 : data_size-fixed_lag;
    int time_step= 0 ;
    Mat current_frame = images.at(time_step).clone(); 
    proposal_filter.initialize(current_frame,estimates.at(time_step));
    //proposal_filter.haar=filter->haar;
    proposal_filter.update_model(theta_x,theta_y);
    for(int k=time_step+1;k<data_size;++k){
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


void pmmh::run_mcmc(){
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    //double forward_filter = filter->getMarginalLikelihood();
    //cout << "----------------" << endl;
    ofstream file1("matrix_pos.txt");
    ofstream file2("matrix_width.txt");
    ofstream file3("matrix_haar_mu.txt");
    ofstream file4("matrix_haar_std.txt");
    ofstream file5("matrix_color.txt");
    ofstream file6("likelihood.txt");
    double forward_filter = marginal_likelihood(theta_x,theta_y);
    for(int n=0;n<mcmc_steps;n++){
        theta_y_prop.clear();
        VectorXd prop_mu=proposal(theta_y[0],100.0);
        theta_y_prop.push_back(prop_mu);
        VectorXd prop_sig=proposal(theta_y[1],10.0);
        prop_sig=prop_sig.array().abs().matrix();
        theta_y_prop.push_back(prop_sig);
        VectorXd prop_color=proposal(theta_y[2],0.1);
        prop_color=prop_color.array().abs().matrix();
        theta_y_prop.push_back(prop_color/prop_color.sum());
        theta_x_prop.clear();
        VectorXd prop_pos=proposal(theta_x[0],1.0);
        prop_pos=prop_pos.array().abs().matrix();
        theta_x_prop.push_back(prop_pos);
        VectorXd prop_std=proposal(theta_x[1],0.1);
        prop_std=prop_std.array().abs().matrix();
        theta_x_prop.push_back(prop_std);
        double proposal_filter = marginal_likelihood(theta_x_prop,theta_y_prop);
        double acceptprob = proposal_filter - forward_filter;
        acceptprob+=gamma_prior(prop_color,SHAPE,SCALE)-gamma_prior(theta_y.at(2),SHAPE,SCALE);
        acceptprob+=gamma_prior(prop_sig,SHAPE,SCALE)-gamma_prior(theta_y.at(1),SHAPE,SCALE);
        acceptprob+=gamma_prior(prop_pos,SHAPE,SCALE)-gamma_prior(theta_x.at(0),SHAPE,SCALE);
        acceptprob+=gamma_prior(prop_std,SHAPE,SCALE)-gamma_prior(theta_x.at(1),SHAPE,SCALE);
        /*cout <<"Theta x:"<< theta_x_prop.at(0).transpose() << endl;
        cout <<"Theta x:"<< theta_x_prop.at(1).transpose() << endl;
        cout <<"Theta y:"<< theta_y_prop.at(0).transpose() << endl;
        cout <<"Theta y:"<< theta_y_prop.at(1).transpose() << endl;
        cout <<"Theta y:"<< theta_y_prop.at(2).transpose() << endl;*/
        double u=unif_rnd(generator);
        if( isfinite(proposal_filter) 
            &&  isfinite(forward_filter) 
            && u < min(1.0,exp(acceptprob)) 
            && (theta_x_prop.at(0).array()>0).all() 
            && (theta_x_prop.at(1).array()>0).all() 
            && (theta_y_prop.at(1).array()>0).all()
            && (theta_y_prop.at(2).array()>0).all()){
            theta_y=theta_y_prop;
            theta_x=theta_x_prop;
            filter->update_model(theta_x,theta_y);
            forward_filter=proposal_filter;
            }
        if(file1.is_open()) file1 << theta_x_prop.at(0).transpose() << endl ;
        if(file2.is_open()) file2 << theta_x_prop.at(1).transpose() << endl ;
        if(file3.is_open()) file3 << theta_y_prop.at(0).transpose() << endl ;
        if(file4.is_open()) file4 << theta_y_prop.at(1).transpose() << endl ;
        if(file5.is_open()) file5 << theta_y_prop.at(2).transpose() << endl ;
        if(file6.is_open()) file6 << forward_filter << endl ;

    }
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();
    file6.close();
}

Rect pmmh::estimate(Mat& image,bool draw){
    Rect estimate=filter->estimate(image,draw);
    estimates.push_back(estimate);
    return estimate;
}

void pmmh::draw_particles(Mat& image){
     filter->draw_particles(image,Scalar(0,255,255));
}
