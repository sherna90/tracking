#include "../include/multinomialnaivebayes.h"

MultinomialNaiveBayes::MultinomialNaiveBayes()
{
    initialized=false;
}

MultinomialNaiveBayes::MultinomialNaiveBayes(MatrixXd &datos,VectorXd &clases)

    X=&datos;
    Y=&clases;
    initialized=get_classes();
}
void MultinomialNaiveBayes::fit()
{
    if(initialized){
        std::map<unsigned int,std::vector<unsigned int>>::iterator iter;
        for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
            classes[iter->first]=Multinomial(iter->second,X , 1.0);
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
bool MultinomialNaiveBayes::get_classes()
{
    for (unsigned int i = 0; i < getY()->rows(); ++i) {
        Xc[(*getY())(i)].push_back(i);
    }

    std::map<unsigned int,std::vector<unsigned int>>::iterator iter;

    for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
        Prior[iter->first]=(double)getXc()[iter->first].size()/getX()->rows();
        //Prior[iter->first]=(double)getXc()[iter->first].size()/getX()rows();
    }
    return true;
}

std::map<unsigned int, std::vector<unsigned int> > MultinomialNaiveBayes::getXc() const
{
    return Xc;
}

void MultinomialNaiveBayes::setXc(const std::map<unsigned int, std::vector<unsigned int> > &value)
{
    Xc = value;
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
