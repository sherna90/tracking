#include "multinomialnaivebayes.hpp"

MultinomialNaiveBayes::MultinomialNaiveBayes()
{
    initialized=false;
}

MultinomialNaiveBayes::MultinomialNaiveBayes(MatrixXd &datos,VectorXd &clases)
{
    X=&datos;
    Y=&clases;
    initialized=true;
}

void MultinomialNaiveBayes::fit(double alpha)
{
    if(initialized)
    {
        #pragma omp parallel
        {
            #pragma omp single
            for (int i = 0; i < getY()->rows(); ++i) {
                if(Xc_sufficient[(*getY())(i)].size()==0)
                {
                    Xc_sufficient[(*getY())(i)]=VectorXd::Zero(X->cols());
                    Prior[(*getY())(i)]=0;
                    classes[(*getY())(i)]=Multinomial();
                }
                Xc_sufficient[(*getY())(i)]+= getX()->row(i);
                Prior[(*getY())(i)]=(Prior[(*getY())(i)]+1.0)/getX()->rows();
                classes[(*getY())(i)].addTheta(Xc_sufficient[(*getY())(i)],alpha);
            }
        }
    }

}

VectorXd MultinomialNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXd c=VectorXd::Zero(Xtest.rows());
    int max_class=0;
    double max_score=-100000000.0;
    double score=0;
    std::map<unsigned int,Multinomial>::iterator iter;
    #pragma omp parallel for private(max_class,max_score,score,iter)
    for (int i = 0; i < Xtest.rows(); ++i) {
        max_class=0;
        max_score=-100000000.0;
        for (iter = classes.begin(); iter != classes.end(); ++iter) {
            score=log(getPrior()[iter->first])+iter->second.log_likelihood(Xtest.row(i));
            if(score > max_score){
                max_score=score;
                max_class=iter->first;
            }
        }
        c(i)=max_class;
    }
    return c;
}

MatrixXd  MultinomialNaiveBayes::get_proba(MatrixXd &Xtest)
{
    MatrixXd proba = MatrixXd::Zero(Xtest.rows(), classes.size());
    //VectorXd log_prob_x = VectorXd::Zero(Xtest.rows());
    if (initialized){
        std::map<unsigned int,Multinomial>::iterator iter;
        #pragma omp parallel for private(iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            for (iter = classes.begin(); iter != classes.end(); ++iter) {
                proba(i, iter->first)=log(getPrior()[iter->first])+iter->second.log_likelihood(Xtest.row(i));
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

std::map<unsigned int, double> MultinomialNaiveBayes::getPrior() const
{
    return Prior;
}

void MultinomialNaiveBayes::setPrior(const std::map<unsigned int, double> &value)
{
    Prior = value;
}
 
MatrixXd *MultinomialNaiveBayes::getX() 
{
    return X;
}

void MultinomialNaiveBayes::setX( MatrixXd *value)
{
    X = value;
}

 VectorXd *MultinomialNaiveBayes::getY() 
{
    return Y;
}

void MultinomialNaiveBayes::setY( VectorXd *value)
{
    Y = value;
}

