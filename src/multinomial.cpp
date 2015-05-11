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
   std::cout<<"counts sum:"<<counts.sum() << " "<<counts.cols()<<std::endl;
   theta=VectorXd(counts.cols());
   #pragma omp parallel for
       for (unsigned int i = 0; i < counts.cols(); ++i) {
           theta(i)=(counts.col(i).sum()+alpha)/total;
           //std::cout<<"sum cols:"<<counts.col(i).sum()<<std::endl;
       }
}

Multinomial::Multinomial(std::vector<unsigned int>  &indices, MatrixXd *X,double alpha)
{
    double sumcols=0.0;
    double sumX=0.0;
    theta=VectorXd(X->cols());
    #pragma omp parallel for reduction(+:sumX)
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        sumX += X->row(indices.at(i)).sum();
    }
    // std::cout<<"sumX:"<<sumX<<std::endl;
    double total = sumX + X->cols()*alpha;
    // std::cout<<"total:"<<total<<std::endl;
    for (unsigned int j = 0; j < X->cols(); ++j) 
    {
        #pragma omp parallel for reduction(+:sumcols)
        for (unsigned int i = 0; i <  indices.size(); ++i)
        {
            sumcols += (*X)(indices[i],j);
        }
        // std::cout<<"sumcols:"<<sumcols<<std::endl;
        theta(j)=(sumcols + alpha)/total;
         // std::cout<<"theta:"<<theta(j)<<std::endl;
        sumcols=0.0;
    }

}

double Multinomial::log_likelihood(VectorXd test)
{
    double log_like=0.0;
    double sum_test=0.0;
    double sum_theta=0.0;
    unsigned int i=0;
//#pragma omp parallel for private(i) reduction(+:sum_test) reduction(+:sum_theta)
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






