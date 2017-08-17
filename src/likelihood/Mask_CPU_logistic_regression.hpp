#ifndef Mask_CPU_LOGISTIC_H
#define Mask_CPU_LOGISTIC_H
#include "logistic_regression.hpp"

class Mask_CPU_LogisticRegression : public LogisticRegression
{
 public:
 	VectorXd train(int n_iter,double alpha=0.01,double tol=0.001);
	void preCompute();
 	VectorXd predict(MatrixXd &_X_test, bool prob=false, bool data_processing = true);
    VectorXd computeGradient();
    void setTrainMask(VectorXd _mask);
 	void setTestMask(VectorXd _mask);
};

#endif
