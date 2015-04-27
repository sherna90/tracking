#include "../include/multinomial.h"

Multinomial::Multinomial()
{

}

Multinomial::Multinomial(MatrixXd &counts)
{
    Multinomial(counts,1.0);
}

Multinomial::Multinomial(MatrixXd &counts, double alpha)
{
    double total=counts.sum()+counts.cols()*alpha;
    theta=VectorXd(counts.cols());
    for (int i = 0; i < counts.cols(); ++i) {
        theta(i)=(counts.col(i).sum()+alpha)/total;
    }
}

double Multinomial::log_likelihood(VectorXd test)
{
    double log_like=0.0;
    VectorXd log_test=VectorXd::Zero(test.size());
    VectorXd log_theta=VectorXd::Zero(test.size());

    for(int i=0;i<test.size();i++){
            log_test[i]=lgamma(test[i]+1);
            log_theta[i]=test[i]*log(this->theta[i]);
    }
    log_like=lgamma(test.sum()+1)-log_test.sum()+log_theta.sum();
    return log_like;
}

VectorXd Multinomial::getTct() const
{
    return Tct;
}

void Multinomial::setTct(const VectorXd &value)
{
    Tct = value;
}
VectorXd Multinomial::getTheta() const
{
    return theta;
}

void Multinomial::setTheta(const VectorXd &value)
{
    theta = value;
}






