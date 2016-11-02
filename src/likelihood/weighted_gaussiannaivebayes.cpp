// Author: Diego Vergara
#include "weighted_gaussiannaivebayes.hpp"

GaussianNaiveBayes::GaussianNaiveBayes()
{
    initialized=true;
}


void GaussianNaiveBayes::fit(MatrixXd &datos,VectorXi &clases,VectorXd weights)
{   
    if (initialized){
        X=&datos;
        Y=&clases;
        int rows = X->rows();
        int cols = X->cols();
        double total_weights = 0.0; // new
        #pragma omp parallel
        {
            #pragma omp single    
            for (int i = 0; i < rows; ++i) {        
                if(means[(*getY())(i)].size()==0){
                    means[(*getY())(i)] = VectorXd::Zero(cols);
                    //ix[(*getY())(i)] = 0;
                    ac_weight[(*getY())(i)] = 0;
                }
                means[(*getY())(i)] += weights(i)*X->row(i);
                //ix[(*getY())(i)] += 1;
                ac_weight[(*getY())(i)]+= weights(i);
                total_weights += weights(i); // new
            }
        }
        #pragma omp parallel
        {
            #pragma omp single    
            for (int i = 0; i < rows; ++i){   
                if(sigmas[(*getY())(i)].size()==0){
                    means[(*getY())(i)] /= ac_weight[(*getY())(i)];
                    sigmas[(*getY())(i)] = VectorXd::Zero(cols);
                }
                sigmas[(*getY())(i)] =  sigmas[(*getY())(i)].array() +((X->row(i).transpose() - means[(*getY())(i)]).array().square() * weights(i)); 
            }
        }  

        std::map<unsigned int,double>::iterator iter;
        //for (iter = ix.begin(); iter != ix.end(); ++iter) {
        for (iter = ac_weight.begin(); iter != ac_weight.end(); ++iter) {                          
            //sigmas[iter->first] /= ac_weight[iter->first];
            sigmas[iter->first] /= iter->second;
            //Prior[iter->first] = (iter->second + 0.0)/rows;
            Prior[iter->first] = (iter->second + 0.0)/total_weights;
        }
    }
}
double GaussianNaiveBayes::log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma){
    double loglike =0.0;
    loglike = -0.5 * ((2*M_PI*sigma).array().log()).sum();
    loglike -= 0.5 * (((data - mean).array().square())/sigma.array()).sum();
    return loglike;
}

VectorXi GaussianNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXi c=VectorXi::Zero(Xtest.rows());
    if (initialized){
        int max_class=0;
        double max_score=-100000000.0;
        double score=0;
        //std::map<unsigned int,int>::iterator iter;
        std::map<unsigned int,double>::iterator iter;
        #pragma omp parallel for private(max_class,max_score,score,iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            max_class=0;
            max_score= -100000000.0;
            //for (iter = ix.begin(); iter != ix.end(); ++iter) {  
            for (iter = ac_weight.begin(); iter != ac_weight.end(); ++iter) {  
                score=log(getPrior()[iter->first]) + log_likelihood(Xtest.row(i), means[iter->first], sigmas[iter->first]);
                if(score > max_score){
                    max_score=score;
                    max_class=iter->first;
                }
            }
            c(i)=max_class;
        }
        return c;
    }
    else{
        return c;
    }

}

MatrixXd GaussianNaiveBayes::get_proba(MatrixXd &Xtest)
{   
    //MatrixXd proba = MatrixXd::Zero(Xtest.rows(), ix.size());
    MatrixXd proba = MatrixXd::Zero(Xtest.rows(), ac_weight.size());
    //VectorXd log_prob_x = VectorXd::Zero(Xtest.rows());
    if (initialized){
        //std::map<unsigned int,int>::iterator iter;
        std::map<unsigned int,double>::iterator iter;
        #pragma omp parallel for private(iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            //for (iter = ix.begin(); iter != ix.end(); ++iter) {  
            for (iter = ac_weight.begin(); iter != ac_weight.end(); ++iter) {  
                proba(i, iter->first)=log(getPrior()[iter->first]) + log_likelihood(Xtest.row(i), means[iter->first], sigmas[iter->first]);
            }
        }
        //log_prob_x = (proba.array().exp()).rowwise().sum().log();
        //for (int i = 0; i < proba.cols(); ++i) proba.col(i) -= log_prob_x;
        double max = proba.maxCoeff();
        double min = proba.minCoeff();
        proba = (proba.array() - min)/(max-min);
        return proba;
    }
    else{
        return proba;
    }

}


std::map<unsigned int, double> GaussianNaiveBayes::getPrior() const
{
    return Prior;
}

void GaussianNaiveBayes::setPrior(const std::map<unsigned int, double> &value)
{
    Prior = value;
}
 MatrixXd *GaussianNaiveBayes::getX() 
{
    return X;
}

void GaussianNaiveBayes::setX( MatrixXd *value)
{
    X = value;
}
 VectorXi *GaussianNaiveBayes::getY() 
{
    return Y;
}

void GaussianNaiveBayes::setY( VectorXi *value)
{
    Y = value;
}

