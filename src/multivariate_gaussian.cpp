#include "../include/multivariate_gaussian.hpp"

MVNGaussian::MVNGaussian(VectorXd _mean, MatrixXd _cov){
    mean = _mean;
    cov = _cov;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

MVNGaussian::MVNGaussian(MatrixXd &data){
    double cols = data.cols();
    mean.resize(cols);
    cov.resize(cols,cols);
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
    VectorXd mvn_sample,mvn_random;
    normal_distribution<double> normal(0.0,1.0);
    for (int i=0;i<mean.size();i++) mvn_random(i)=normal(generator);
    LLT<MatrixXd> cholSolver(cov);
    MatrixXd upperL = cholSolver.matrixL();
    mvn_sample= upperL*mvn_random+ mean;
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


// int main(){
//     MatrixXd m(14,4);
//     m << 1359, 413, 120, 362,
//          584, 446, 84, 256,
//          729, 457, 39, 119,
//          460, 442,  90, 274,
//          643, 461,  59, 180,
//          1494, 408, 112 ,338,
//          572, 364 ,128, 388,
//          1097, 433,  39, 119,
//          1324, 258, 182, 550,
//          1021, 436,  31,  96,
//          549, 476 , 27,  83,
//          545, 457,  39, 119,
//          1254, 446 , 33, 103,
//          935, 429 , 41, 127;
    
//     VectorXd mean(4);
//     mean << 1193, 989,  20,  50;

//     MatrixXd cov = POSITION_LIKELIHOOD_STD*POSITION_LIKELIHOOD_STD*MatrixXd::Identity(4, 4);


//     /* Testing getter and setter */
//     MVNGaussian element (m);
//     MVNGaussian element(mean, cov);
//     cout << "Mean: \n" << element.getMean() << "\n\n";
//     cout << "Covariance Matrix: \n" << element.getCov() << "\n\n";
//     cout << "Log Likelihood: \n" << element.log_likelihood(m) << endl;
//     return 0;
// }