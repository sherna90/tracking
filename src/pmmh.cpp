#include "../include/pmmh.hpp"

const float SHAPE=0.1;
const float SCALE=0.1;
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
    images = vector<Mat>();
    std::gamma_distribution<double> prior(SHAPE,SCALE);
    filter=new particle_filter(n_particles);
    filter->initialize(current_frame,ground_truth);
    images.push_back(current_frame);
    reference_roi=ground_truth;
    theta_y.resize(2);
    theta_y(0)=1.0/prior(generator);
    theta_y(1)=1.0/prior(generator);
    theta_x.resize(2);
    theta_x(0)=1.0/prior(generator);
    theta_x(1)=1.0/prior(generator);
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

void pmmh::predict(){
   /* normal_distribution<double> position_random_walk(0.0,theta_x(0));
    normal_distribution<double> scale_random_walk(0.0,theta_x(1));
    if(initialized==true){
        vector<particle> tmp_new_states;
        float _x=0.0,_y=0.0,_x_p=0.0,_y_p=0.0,_width=0.0,_height=0.0,_width_p=0.0,_height_p=0.0;
        for (int i=0;i<n_particles;i++){
            particle state=filter->states[i];
            _x+=float(1.0/n_particles)*state.x;
            _y+=float(1.0/n_particles)*state.y;
            _x_p+=float(1.0/n_particles)*state.x_p;
            _y_p+=float(1.0/n_particles)*state.y_p;
            _width+=(state.width > 0 ) ? float(1.0/n_particles)*state.width : float(1.0/n_particles)*reference_roi.width; 
            _height+=(state.height > 0 ) ? float(1.0/n_particles)*state.height : float(1.0/n_particles)*reference_roi.height;   
            _width_p+=(state.width_p > 0 ) ? float(1.0/n_particles)*state.width_p : float(1.0/n_particles)*reference_roi.width; 
            _height_p+=(state.height_p > 0 ) ? float(1.0/n_particles)*state.height_p : float(1.0/n_particles)*reference_roi.height;
        }
        for (int i=0;i<n_particles;i++){  
            particle state;
            float _dx,_dy,_dw,_dh;
            _dx=(_x-_x_p);
            _dy=(_y-_y_p);
            _dw=(_width-_width_p);
            _dh=(_height-_height_p);
            state.x=MAX(cvRound(_x+_dx+position_random_walk(generator)),0);
            state.y=MAX(cvRound(_y+_dy+position_random_walk(generator)),0);
            state.x_p=_x;
            state.y_p=_y;
            state.width=MAX(cvRound(_width+_dw+scale_random_walk(generator)),0);
            state.height=MAX(cvRound(_height+_dh+scale_random_walk(generator)),0);
            state.width_p=cvRound(_width);
            state.height_p=cvRound(_height);
            state.scale=1.0f+scale_random_walk(generator);
            state.scale_p=0.0;
            cout << "x:" << state.x << ",y:" << state.y << ",x_p:" << state.x_p << ",y_p:" << state.y_p  <<",w:" << state.width <<",h:" << state.height << state.y <<",dw:" << _dw <<",dh:" << _dh << endl;          
            tmp_new_states.push_back(state);
        }
        filter->states.swap(tmp_new_states);
        tmp_new_states = vector<particle>();
    }*/
    filter->predict();
}

double pmmh::marginal_likelihood(VectorXd theta_x,VectorXd theta_y){
    particle_filter pmmh(n_particles);
    Mat current_frame = images[0].clone(); 
    pmmh.initialize(current_frame,reference_roi);
    pmmh.update_model(theta_x,theta_y);
    int time_step=MIN((int)images.size(),fixed_lag);
    double res=0.0;
    //cout << "time:";
    for(int k=0;k<time_step;++k){
        //cout << k << "," ;
        current_frame = images[k].clone();
        pmmh.predict();
        //pmmh.resample();
        pmmh.update(current_frame);
    }
    //cout << endl;
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

void pmmh::update(Mat& current_frame){
    images.push_back(current_frame);
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    filter->update(current_frame);
    //double forward_filter = filter->getMarginalLikelihood();
    double forward_filter = marginal_likelihood(theta_x,theta_y);
    for(int n=0;n<mcmc_steps;n++){
        theta_y_prop=continuous_proposal(theta_y);
        theta_x_prop=continuous_proposal(theta_x);
        double proposal_filter = marginal_likelihood(theta_x_prop,theta_y_prop);
        double acceptprob = proposal_filter - forward_filter;
        acceptprob+=igamma_prior(theta_y_prop,SHAPE,SCALE)-igamma_prior(theta_y,SHAPE,SCALE);
        double u=unif_rnd(generator);
        if( isfinite(proposal_filter) &&  isfinite(forward_filter) 
            && log(u) < acceptprob && (theta_y_prop.array()>0).all() 
            && (theta_x_prop.array()>0).all()){
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
     filter->draw_particles(image,Scalar(0,255,255));
}