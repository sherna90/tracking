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
        std::map<int,std::vector<int> >::iterator iter;
        for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
            int nc=iter->second.size();
            MatrixXd Xtr(nc,getX()->cols());
            for (int i = 0; i < nc; ++i) {
                Xtr.row(i)=getX()->row(iter->second[i]);
            }
            classes[iter->first]=Multinomial(Xtr,1.0);
        }
    }
}

VectorXi MultinomialNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXi c=VectorXi::Zero(Xtest.rows());
    //std::ofstream likelihood_txt("likelihood.txt");
    // original...

    for (long i = 0; i < Xtest.rows(); ++i) {
        std::map<int,Multinomial>::iterator iter;
        double max_class=0.0;
        double max_score=-100000000.0;
        for (iter = classes.begin(); iter != classes.end(); ++iter) {
            double score=log(getPrior()[iter->first])+iter->second.log_likelihood(std::move(Xtest.row(i)));
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
    for (long i = 0; i < getY()->rows(); ++i) {
        Xc[(*getY())(i)].push_back(i);
    }

    std::map<int,std::vector<int> >::iterator iter;
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
std::map<int, std::vector<int> > MultinomialNaiveBayes::getXc() const
{
    return Xc;
}

void MultinomialNaiveBayes::setXc(const std::map<int, std::vector<int> > &value)
{
    Xc = value;
}
std::map<int, double> MultinomialNaiveBayes::getPrior() const
{
    return Prior;
}

void MultinomialNaiveBayes::setPrior(const std::map<int, double> &value)
{
    Prior = value;
}





