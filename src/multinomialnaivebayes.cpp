#include "../include/multinomialnaivebayes.h"

MultinomialNaiveBayes::MultinomialNaiveBayes()
{
    initialized=false;
}

MultinomialNaiveBayes::MultinomialNaiveBayes(MatrixXd &datos,VectorXi &clases)
{
    X=&datos;
    Y=&clases;
    initialized=get_classes();
}
void MultinomialNaiveBayes::fit()
{
    if(initialized){
        std::map<long,std::vector<long>>::iterator iter;
        for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
            long nc=iter->second.size();
            MatrixXd Xtr(nc,getX()->cols());
            for (long i = 0; i < nc; ++i) {
                Xtr.row(i)=getX()->row(iter->second[i]);
            }
            classes[iter->first]=Multinomial(Xtr,1.0);
        }
    }
}

VectorXi MultinomialNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXi c=VectorXi::Zero(Xtest.rows());
    double max_class=0.0;
    double max_score=-100000000.0;
    double score=0;
    std::map<long,Multinomial>::iterator iter;
#pragma omp parallel for private(max_class,max_score,score,iter)
    for (long i = 0; i < Xtest.rows(); ++i) {
        max_class=0.0;
        max_score=-100000000.0;
        for (iter = classes.begin(); iter != classes.end(); ++iter) {
           score=log(getPrior()[iter->first])+iter->second.log_likelihood(std::move(Xtest.row(i)));
            if(score > max_score){
                max_score=score;
                max_class=iter->first;
            }
            score=0;
        }
        c(i)=max_class;
    }
    return c;

}
bool MultinomialNaiveBayes::get_classes()
{
    for (long i = 0; i < getY()->rows(); ++i) {
        Xc[(*getY())(i)].push_back(i);
    }

    std::map<long,std::vector<long>>::iterator iter;

    for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
        Prior[iter->first]=(double)getXc()[iter->first].size()/getX()->rows();
    }
    return true;
}
const MatrixXd *MultinomialNaiveBayes::getX() const
{
    return X;
}

void MultinomialNaiveBayes::setX(const MatrixXd *value)
{
    X = value;
}
const VectorXi *MultinomialNaiveBayes::getY() const
{
    return Y;
}

void MultinomialNaiveBayes::setY(const VectorXi *value)
{
    Y = value;
}
std::map<long, std::vector<long> > MultinomialNaiveBayes::getXc() const
{
    return Xc;
}

void MultinomialNaiveBayes::setXc(const std::map<long, std::vector<long> > &value)
{
    Xc = value;
}
std::map<long, double> MultinomialNaiveBayes::getPrior() const
{
    return Prior;
}

void MultinomialNaiveBayes::setPrior(const std::map<long, double> &value)
{
    Prior = value;
}





