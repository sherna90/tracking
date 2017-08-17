#ifndef CPU_LOGISTIC_H
#define CPU_LOGISTIC_H
#include "logistic_regression.hpp"

class CPU_LogisticRegression : public LogisticRegression
{
 public:
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
	void preCompute();
 	VectorXd predict(MatrixXd &_X_test, bool prob=false, bool data_processing = true);
    VectorXd computeGradient();
};

#endif
