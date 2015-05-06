#include "../include/multinomialnaivebayes.h"

MultinomialNaiveBayes::MultinomialNaiveBayes()
{
    initialized=false;
}

MultinomialNaiveBayes::MultinomialNaiveBayes(MatrixXd &datos,VectorXi &clases)
{
//    X=&datos;
//    Y=&clases;
    X=&datos;
    Y=&clases;
    initialized=get_classes();
}
void MultinomialNaiveBayes::fit()
{
    if(initialized){
       // std::map<long,std::vector<long>>::iterator iter;
       // for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
       //     long nc=iter->second.size();
       //     MatrixXd Xtr(nc,getX()->cols());
       //     #pragma omp parallel for
       //     for (long i = 0; i < nc; ++i) {
       //         Xtr.row(i)=getX()->row(iter->second[i]);
       //     }
       //     classes[iter->first]=Multinomial(Xtr,1.0);
       // }

        //nuevooo
        std::map<long,std::vector<long>>::iterator iter;
        for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
            classes[iter->first]=Multinomial(iter->second,X , 1.0);
        }
    }
}

VectorXi MultinomialNaiveBayes::test(MatrixXd &Xtest)
{
    VectorXi c=VectorXi::Zero(Xtest.rows());

    int corte=0;
    if(Xtest.rows()%2==0)
        corte=Xtest.rows()/2;
    else
        corte=(Xtest.rows()+1)/2;


    //omp_set_nested(1);
    #pragma omp  sections
    {
        #pragma omp section
        {
            doTest(0,corte,c,Xtest);
        }
        #pragma omp section
        {
            doTest(corte,Xtest.rows(),c,Xtest);
        }

    }
    return c;
}
bool MultinomialNaiveBayes::get_classes()
{
    for (long i = 0; i < getY()->rows(); ++i) {
        Xc[(*getY())(i)].push_back(i);
        //Xc[(getY())(i)].push_back(i);
    }

    std::map<long,std::vector<long>>::iterator iter;

    for (iter = Xc.begin(); iter != Xc.end(); ++iter) {
        Prior[iter->first]=(double)getXc()[iter->first].size()/getX()->rows();
        //Prior[iter->first]=(double)getXc()[iter->first].size()/getX()rows();
    }
    return true;
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

void MultinomialNaiveBayes::doTest(long a, long b,VectorXi &c,MatrixXd &Xtest)
{
    double max_class=0.0;
    double max_score=-100000000.0;
    double score=0;
    std::map<long,Multinomial>::iterator iter;
    #pragma omp parallel for private(max_class,max_score,score,iter)
    for (long i = a; i < b; ++i) {
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










