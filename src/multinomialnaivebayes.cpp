#include "../include/multinomialnaivebayes.h"

MultinomialNaiveBayes::MultinomialNaiveBayes()
{
    initialized=false;
}

MultinomialNaiveBayes::MultinomialNaiveBayes(MatrixXd &datos,VectorXi &clases)
{
    X=&datos;
    Y=&clases;
    initialized=true;
}
void MultinomialNaiveBayes::fit()
{
    if(initialized)
    {
        for (unsigned int i = 0; i < getY()->rows(); ++i) {
            if(Xc_sufficient[(*getY())(i)].size()==0)
            {
                Xc_sufficient[(*getY())(i)]=VectorXd::Zero(X->cols());
                Prior[(*getY())(i)]=0;
                classes[(*getY())(i)]=Multinomial();
            }
            Xc_sufficient[(*getY())(i)]+=getX()->row(i);
            Prior[(*getY())(i)]=(Prior[(*getY())(i)]+1.0)/getX()->rows();
            classes[(*getY())(i)].addTheta(Xc_sufficient[(*getY())(i)] ,1.0);
        }
    }
}

VectorXd MultinomialNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXd c=VectorXd::Zero(Xtest.rows());
    double max_class=0.0;
    double max_score=-100000000.0;
    double score=0;
    std::map<unsigned int,Multinomial>::iterator iter;
    #pragma omp parallel for private(max_class,max_score,score,iter)
    for (unsigned int i = 0; i < Xtest.rows(); ++i) {
        max_class=0.0;
        max_score=-100000000.0;
        for (iter = classes.begin(); iter != classes.end(); ++iter) {
            score=log(getPrior()[iter->first])+iter->second.log_likelihood(std::move(Xtest.row(i)));
            if(score > max_score){
                max_score=score;
                max_class=iter->first;
            }
        }
        c(i)=max_class;
    }
    return c;
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
 VectorXi *MultinomialNaiveBayes::getY() 
{
    return Y;
}

void MultinomialNaiveBayes::setY( VectorXi *value)
{
    Y = value;
}
