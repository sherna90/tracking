//Author: Diego Vergara
#include "incremental_gaussiannaivebayes.hpp"

GaussianNaiveBayes::GaussianNaiveBayes()
{
    initialized=false;
}

GaussianNaiveBayes::GaussianNaiveBayes(MatrixXd &datos,VectorXd &clases)
{
    X=&datos;
    Y=&clases;
    initialized = true;
    one_fit = false;
    Cols = 0;
    Rows = 0;
}

void GaussianNaiveBayes::fit()
{
    if (initialized and !one_fit){
        partial_fit((*getX()),(*getY()), 0.0);
        //one_fit = true; 
    }
    else{
        cout << "Error: Model not initialized or previously fitted" << endl;
    }

}


void GaussianNaiveBayes::partial_fit(MatrixXd &datos,VectorXd &clases, double learning_rate)
{   
    X=&datos;
    Y=&clases;
    int new_rows = getX()->rows();
    int new_cols = getX()->cols();
    if (initialized){
        
        std::map<unsigned int,VectorXd> new_means, new_sigmas;
        std::map<unsigned int,double> new_Prior;
        #pragma omp parallel
        {
            #pragma omp single    
            for (int i = 0; i < new_rows; ++i) {        
                if(new_means[(*getY())(i)].size()==0){
                    new_means[(*getY())(i)] = VectorXd::Zero(new_cols);
                    new_Prior[(*getY())(i)] = 0.0;
                }
                new_means[(*getY())(i)] += getX()->row(i);
                new_Prior[(*getY())(i)] += 1.0;
            }
        }
        #pragma omp parallel
        {
            #pragma omp single    
            for (int i = 0; i < new_rows; ++i){   
                if(new_sigmas[(*getY())(i)].size()==0){
                    new_means[(*getY())(i)] /= new_Prior[(*getY())(i)];
                    new_Prior[(*getY())(i)] /= new_rows;
                    new_sigmas[(*getY())(i)] = VectorXd::Zero(new_cols);
                }
                new_sigmas[(*getY())(i)] =  new_sigmas[(*getY())(i)].array() +(getX()->row(i).transpose() - new_means[(*getY())(i)]).array().square(); 
            }
        } 
        std::map<unsigned int,double>::iterator iter;
        for (iter = new_Prior.begin(); iter != new_Prior.end(); ++iter) {  
            new_sigmas[iter->first] /= new_Prior[iter->first] * new_rows; 
        }

        if (!one_fit)
        {
            Means = new_means;
            Sigmas = new_sigmas;
            Prior = new_Prior;
            Cols = new_cols;
            Rows = new_rows;
            one_fit = true;
            //throw std::exception();
        }
        else{
            // Update model
            if (Cols == new_cols){
                std::map<unsigned int,double>::iterator iter;
                for (iter = new_Prior.begin(); iter != new_Prior.end(); ++iter) {  
                    if(Means[iter->first].size()==0){
                            Means[iter->first] = VectorXd::Zero(new_cols);
                            Sigmas[iter->first] = VectorXd::Zero(new_cols);
                            Prior[iter->first] = 0.0;
                    }
                    Means[iter->first] =  ((1-learning_rate)*Means[iter->first]*Prior[iter->first]*Rows + (learning_rate)*new_means[iter->first]*iter->second*new_rows) 
                                            / (Prior[iter->first]*Rows + iter->second*new_rows);
                    /*Means[iter->first] =  (Means[iter->first]*Prior[iter->first]*Rows + new_means[iter->first]*iter->second*new_rows) 
                                            / (Prior[iter->first]*Rows + iter->second*new_rows);*/

                    Sigmas[iter->first] = ((1-learning_rate)*(Sigmas[iter->first]*Prior[iter->first]*Rows + (learning_rate)*new_sigmas[iter->first]*iter->second*new_rows).array() 
                                            + ((Prior[iter->first]*Rows)/ ((Prior[iter->first]*Rows + iter->second*new_rows)*iter->second*new_rows))
                                            * (learning_rate*(1-learning_rate))*(Means[iter->first]*iter->second*new_rows - (learning_rate)*new_means[iter->first]*iter->second*new_rows).array().square())
                                            / (Prior[iter->first]*Rows + iter->second*new_rows);

                    /*Sigmas[iter->first] = ((Sigmas[iter->first]*Prior[iter->first]*Rows + new_sigmas[iter->first]*iter->second*new_rows).array() 
                                            + ((Prior[iter->first]*Rows)/ ((Prior[iter->first]*Rows + iter->second*new_rows)*iter->second*new_rows))
                                            * (Means[iter->first]*iter->second*new_rows - new_means[iter->first]*iter->second*new_rows).array().square())
                                            / (Prior[iter->first]*Rows + iter->second*new_rows);*/


                    Prior[iter->first] = (Prior[iter->first]*Rows + iter->second*new_rows)/(Rows+new_rows);
                }
                Rows+=new_rows;
            }
            else{
                cout << "Error: Inconsistent data (colums size)" << endl;
            }
        }
        
    }
    else{
        cout << "Error: Model not initialized" << endl;
    }

}
double GaussianNaiveBayes::log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma){
    double loglike =0.0;
    double eps = std::numeric_limits<double>::epsilon();
    loglike = -0.5 * (((2*M_PI*sigma).array()+eps).log()).sum();
    loglike -= 0.5 * (((data - mean).array().square())/(sigma.array()+eps)).sum();
    return loglike;
}

double GaussianNaiveBayes::likelihood(VectorXd data, VectorXd mean, VectorXd sigma){
    double likelihood =0.0;
    double eps = std::numeric_limits<double>::epsilon();
    likelihood = ((-((data - mean).array().square())/(2*sigma.array()+eps)).exp() / (2*M_PI*sigma).array().square()).prod();
    return likelihood;
}

VectorXd GaussianNaiveBayes::predict(MatrixXd &Xtest)
{
    VectorXd c=VectorXd::Zero(Xtest.rows());
    if (initialized){
        int max_class=0;
        double max_score=-100000000.0;
        double score=0;
        std::map<unsigned int,double>::iterator iter;
        #pragma omp parallel for private(max_class,max_score,score,iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            max_class=0;
            max_score= -100000000.0;
            for (iter = Prior.begin(); iter != Prior.end(); ++iter) {  
                score=log(iter->second) + log_likelihood(Xtest.row(i), Means[iter->first], Sigmas[iter->first]);
                //score= log_likelihood(Xtest.row(i), Means[iter->first], Sigmas[iter->first]);
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
        cout << "Error: Model not initialized or not previously fitted" << endl;
        return c;
    }

}

MatrixXd GaussianNaiveBayes::get_proba(MatrixXd &Xtest)
{   
    MatrixXd proba = MatrixXd::Zero(Xtest.rows(), Prior.size());
    if (initialized){
        std::map<unsigned int,double>::iterator iter;
        #pragma omp parallel for private(iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            for (iter = Prior.begin(); iter != Prior.end(); ++iter) {  
                //proba(i, iter->first)=log(iter->second) + log_likelihood(Xtest.row(i), Means[iter->first], Sigmas[iter->first]);
                proba(i, iter->first)=log_likelihood(Xtest.row(i), Means[iter->first], Sigmas[iter->first]);
            }
        }
        //double max = proba.maxCoeff();
        //double min = proba.minCoeff();
        //proba = (proba.array() - min)/(max-min);
        return proba;
    }
    else{
        cout << "Error: Model not initialized or not previously fitted" << endl;
        return proba;
    }

}

VectorXd GaussianNaiveBayes::predict_proba(MatrixXd &Xtest, int target)
{   
    MatrixXd proba = MatrixXd::Zero(Xtest.rows(), Prior.size());
    MatrixXd normalization_const = MatrixXd::Zero(Xtest.rows(), Prior.size());
    VectorXd log_sum_exp = VectorXd::Zero(Xtest.rows());
    if (initialized){
        std::map<unsigned int,double>::iterator iter;
        #pragma omp parallel for private(iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            for (iter = Prior.begin(); iter != Prior.end(); ++iter) {  
                proba(i, iter->first)=log_likelihood(Xtest.row(i), Means[iter->first], Sigmas[iter->first]);

            }
        }
        VectorXd max_val=proba.rowwise().maxCoeff();
        normalization_const = proba.colwise()-max_val;
        log_sum_exp= proba.col(target).array() - normalization_const.array().exp().rowwise().sum();
        return log_sum_exp;
    }
    else{
        cout << "Error: Model not initialized or not previously fitted" << endl;
        return log_sum_exp;
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
 VectorXd *GaussianNaiveBayes::getY() 
{
    return Y;
}

void GaussianNaiveBayes::setY( VectorXd *value)
{
    Y = value;
}

