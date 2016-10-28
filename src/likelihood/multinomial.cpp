#include "multinomial.hpp"

Multinomial::Multinomial()
{

}
Multinomial::Multinomial(VectorXd &theta)
{

    setTheta(theta);
}

Multinomial::Multinomial(MatrixXd &counts, double &alpha)
{
    sufficient=VectorXd(counts.cols());
    double total=counts.sum()+counts.cols()*alpha;
    // std::cout<<"counts sum:"<<counts.sum() << " "<<counts.cols()<<std::endl;
    theta=VectorXd(counts.cols());
    //#pragma omp parallel for
    for (int i = 0; i < counts.cols(); ++i) {
         theta(i)=(counts.col(i).sum()+alpha)/total;
         sufficient(i)=counts.col(i).sum();
    }
}

Multinomial::Multinomial(VectorXd &sufficient,double &alpha)
{
    this->sufficient=VectorXd(sufficient.size());
    addTheta(sufficient,alpha);

}

double Multinomial::log_likelihood(const VectorXd &test)
{
    double log_like=0.0;
    double sum_test=0.0;
    double sum_theta=0.0;
    for(int i=0;i<test.size();i++){
        sum_test+=lgamma(test[i]+1);
        sum_theta+= (this->theta[i]!=0.0) ? test[i]*log(this->theta[i]) : 0.0;
    }
    log_like=lgamma(test.sum()+1)-sum_test+sum_theta;
    return log_like;
}

VectorXd Multinomial::getTheta() const
{
    return theta;
}

void Multinomial::setTheta(const VectorXd &value)
{
    theta = value;
}
void Multinomial::addTheta(VectorXd &value,double &alpha)
{
    if(sufficient.size()==0)
        this->sufficient=VectorXd(value.size());
    sufficient+=value;
    theta= (sufficient.array()+alpha);
    theta/=(sufficient.sum() +value.cols()*alpha);
}
