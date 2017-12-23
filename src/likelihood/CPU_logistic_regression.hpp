#ifndef CPU_LOGISTIC_H
#define CPU_LOGISTIC_H
#include "logistic_regression.hpp"

class CPU_LogisticRegression : public LogisticRegression
{
 public:
 	VectorXd train(int n_iter,int mini_batch,double alpha=0.01,double step_size=0.001);
	VectorXd predict(MatrixXd &_X_test, bool prob=false, bool data_processing = true);  
	 
 private:
	VectorXd computeGradient(int iter,int mini_batch);
	void preCompute(int iter,int mini_batch);
};

#endif
