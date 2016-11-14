#include "multivariate_gaussian.hpp"

MVNGaussian::MVNGaussian(VectorXd _mean, MatrixXd _cov){
    dim=_mean.size();
    mean = _mean;
    cov = _cov;
    //cout <<  "MVN Random  m:" << mean << ",cov : "<< cov  << endl;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

MVNGaussian::MVNGaussian(MatrixXd &data){
    dim = data.cols();
    mean.resize(dim);
    cov.resize(dim,dim);
    MatrixXd centered;

    /* Getting mean for every column */
    mean = data.colwise().mean();
    
    /* Covariance Matrix */
    centered = data.rowwise() - mean.transpose();
    cov = (centered.adjoint() * centered) / double(data.rows() - 1);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

VectorXd MVNGaussian::getMean(void){
    return mean;
}

MatrixXd MVNGaussian::getCov(void){
    return cov;
}

void MVNGaussian::setMean(VectorXd _mean){
    mean = _mean;
}

void MVNGaussian::setCov(MatrixXd _cov){
    cov = _cov;
}

VectorXd MVNGaussian::sample(){
    VectorXd mvn_sample=VectorXd::Zero(dim);
    VectorXd mvn_random=VectorXd::Random(dim);
    normal_distribution<double> normal(0.0,1.0);
    for (int i=0;i<dim;i++) mvn_random(i)=normal(generator);
    LLT<MatrixXd> cholSolver(cov);       
    MatrixXd upperL = cholSolver.matrixL();
    mvn_sample= upperL*mvn_random+ mean;
    return mvn_sample;
}

MatrixXd MVNGaussian::sample(int n_samples){
    MatrixXd mvn_sample,mvn_random;
    srand((unsigned int) time(0));
    mvn_random=MatrixXd::Zero(n_samples,dim);
    mvn_random.noalias() = mvn_random.unaryExpr([](double elem) // changed type of parameter
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        mt19937 generator(seed);
        normal_distribution<double> normal(0.0,1.0);
        elem=normal(generator);
        return elem;
    });
    LLT<MatrixXd> cholSolver(cov);
    MatrixXd upperL = cholSolver.matrixL();
    mvn_sample= mvn_random*upperL;
    mvn_sample.rowwise()+=mean.transpose();
    return mvn_sample;
}

VectorXd MVNGaussian::log_likelihood(MatrixXd data){
    double rows = data.rows();
    double cols = data.cols();
    VectorXd loglike = VectorXd::Zero(rows);
    /* Getting inverse matrix for 'cov' with Cholesky */
    LLT<MatrixXd> chol(cov);
    MatrixXd L = chol.matrixL();
    MatrixXd cov_inverse = L.adjoint().inverse() * L.inverse();
    double logdet=log(cov.determinant());
     for(unsigned i=0;i<rows;i++){
        VectorXd tmp1 = data.row(i);
        tmp1 -= mean;
        MatrixXd tmp2 = tmp1.transpose() * cov_inverse;
        tmp2 = tmp2 * tmp1;
        loglike(i) = -0.5 * tmp2(0,0) - (cols/2) * log(2*M_PI) -(0.5) * logdet;
    }
    //cout << loglike << endl;
    return loglike;
}
