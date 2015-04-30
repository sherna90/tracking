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
        for (long i = 0; i < counts.cols(); ++i) {
            theta(i)=(counts.col(i).sum()+alpha)/total;
            //std::cout <<"Tct:"<<counts.col(i).sum()<<std::endl;
        }
}

double Multinomial::log_likelihood(VectorXd test)
{
    double log_like=0.0;
    double sum_test=0.0;
    double sum_theta=0.0;
    long i=0;
#pragma omp parallel for private(i) reduction(+:sum_test) reduction(+:sum_theta)
    for(i=0;i<test.size();i++){
        sum_test+=lgamma(test[i]+1);
        sum_theta+=test[i]*log(this->theta[i]);
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






