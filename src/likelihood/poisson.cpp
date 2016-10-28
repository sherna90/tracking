#include "poisson.hpp"

Poisson::Poisson()
{
    sample_size=0.0;
}
Poisson::Poisson(VectorXd &lambda)
{
    setLambda(lambda);
    sample_size=0.0;
}

double Poisson::log_likelihood(const VectorXd &test)
{
    double log_like=0.0;
    double sum_test=0.0;
    double sum_lambda=0.0;
    unsigned int i=0;
    //#pragma omp parallel for private(i) reduction(+:sum_test) reduction(+:sum_lambda)
    for(i=0;i<test.size();i++){
        sum_test+=lgamma(test[i]+1)+this->lambda[i];
        sum_lambda+=test[i]*log(this->lambda[i]);
    }
    log_like=sum_lambda-sum_test;
    return log_like;
}

VectorXd Poisson::getLambda() const
{
    return lambda;
}

void Poisson::setLambda(const VectorXd &value)
{
    lambda = value;
}

void Poisson::addLambda(VectorXd &value)
{
    if(sufficient.size()==0)
        this->sufficient=VectorXd(value.size());
    sufficient+=value;
    sample_size++;
    lambda= sufficient/sample_size;
}
