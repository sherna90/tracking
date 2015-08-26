/**
 * @file pmmh.cpp
 * @brief particle marginal metropolis hastings 
 * @author Sergio Hernandez
 */

#include "../include/pmmh.hpp"


PMMH::PMMH(vector<Mat>& _images, vector<string>& _gt_vect){
    images=_images;
    gt_vect=_gt_vect;
}

double PMMH::marginal_likelihood(int num_particles,int time_step,int fixed_lag,VectorXd theta_x,VectorXd theta_y){
    particle_filter pmmh_filter(num_particles);
    int start_time;
    (fixed_lag==0) || (time_step<fixed_lag) ? start_time=0 : start_time=time_step-fixed_lag;
    for(int k=start_time;k<=time_step;k++){
        Mat current_frame = images[k].clone();
        string current_gt = gt_vect[k];
        if(!pmmh_filter.is_initialized()){
            Rect ground_truth=updateGroundTruth(current_frame,current_gt,false);
            pmmh_filter.initialize(current_frame,ground_truth);
            pmmh_filter.update_model(theta_x,theta_y);
        }
        else if(pmmh_filter.is_initialized()){
            pmmh_filter.predict();
            pmmh_filter.update_discrete(current_frame);
        }
    }
    return pmmh_filter.getMarginalLikelihood();
}

VectorXd PMMH::discrete_proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    double eps= std::numeric_limits<double>::epsilon();
    for(int i=0;i<alpha.size();i++){
        gamma_distribution<double> color_prior(alpha[i],0.1);
        double val=color_prior(generator);
        proposal[i] = (val>0.0) ? val : eps;
    }
    proposal.normalize();
    return proposal;
}

VectorXd PMMH::continuous_proposal(VectorXd alpha){
    VectorXd proposal(alpha.size());
    for(int i=0;i<alpha.size();i++){
        normal_distribution<double> random_walk(alpha[i],0.1);
        double val=random_walk(generator);
        proposal[i] = val;
    }
    return proposal;
}

double PMMH::gamma_prior(VectorXd x, VectorXd a, double b)
{
    double loglike=0.0;
    for(int i=0;i<a.size();i++){
        if (x(i) >= 0 || a(i) >= 0 || b >= 0){
            loglike+=-x(i)*b+(a(i)-1.0)*log(x(i))+a(i)*log(b)-lgamma(a(i));
        }
    }
    return loglike;
}

void PMMH::run(int num_particles,int fixed_lag,int mcmc_steps){
    particle_filter filter(num_particles);
    int num_frames=(int)images.size();
    Performance track_algorithm;
    Performance particle_filter_algorithm;
    VectorXd theta_x,theta_x_prop,theta_y,theta_y_prop,alpha;
    alpha.setOnes((int)H_BINS*S_BINS);
    //alpha.normalize();
    dirichlet prior=dirichlet(alpha);
    Gaussian pos_prior=Gaussian(0.0,POS_STD);
    Gaussian vel_prior=Gaussian(0.0,VEL_STD);
    Gaussian scale_prior=Gaussian(0.0,SCALE_STD);
    uniform_real_distribution<double> unif_rnd(0.0,1.0);
    for(int k=0;k <num_frames;++k){
        Mat current_frame = images[k].clone();
        string current_gt = gt_vect[k];
        Rect ground_truth=updateGroundTruth(current_frame,current_gt,true);
        if(!filter.is_initialized()){
            filter.initialize(current_frame,ground_truth);
            theta_y=filter.get_discrete_model();
            theta_x=filter.get_continuous_model();
        }
        else if(filter.is_initialized()){
            filter.predict();
            filter.update_discrete(current_frame);
            if(k>fixed_lag){
            double forward_filter = marginal_likelihood(num_particles,k,fixed_lag,theta_x,theta_y);
            for(int n=0;n<mcmc_steps;n++){
                theta_y_prop=discrete_proposal(theta_y);
                theta_x_prop=continuous_proposal(theta_x);
                double proposal_filter = marginal_likelihood(num_particles,k,fixed_lag,theta_x_prop,theta_y_prop);
                double acceptprob = proposal_filter - forward_filter;
                acceptprob+=prior.log_likelihood(theta_y_prop)-prior.log_likelihood(theta_y);
                acceptprob+=pos_prior.log_likelihood(theta_x_prop(0))-pos_prior.log_likelihood(theta_x(0));
                acceptprob+=vel_prior.log_likelihood(theta_x_prop(1))-vel_prior.log_likelihood(theta_x(1));
                acceptprob+=scale_prior.log_likelihood(theta_x_prop(2))-scale_prior.log_likelihood(theta_x(2));
                double u=unif_rnd(generator);
                if( log(u) < acceptprob){
                    theta_y=theta_y_prop;
                    theta_x=theta_x_prop;
                    filter.update_model(theta_x_prop,theta_y_prop);
                    forward_filter=proposal_filter;
                    }
                }
            }
        }
        Rect estimate=filter.estimate(current_frame,true);
        double r1=particle_filter_algorithm.calc(ground_truth,estimate);
        if(r1<0.1) filter.reinitialize();
    }
    cout << "PMMH algorithm >> " <<"average precision:" << particle_filter_algorithm.get_avg_precision()/num_frames << ",average recall:" << particle_filter_algorithm.get_avg_recall()/num_frames << endl;
}