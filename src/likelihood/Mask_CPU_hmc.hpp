//Author: Diego Vergara
#ifndef Mask_CPU_HAMILTONIAN_MC_H
#define Mask_CPU_HAMILTONIAN_MC_H
#include "Mask_CPU_logistic_regression.hpp"
#include "hmc.hpp"

class Mask_CPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, double _step_size = 0.01, int _num_step = 100, double _path_lenght = 0.0);
	void run(bool warmup_flag = false);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 0, bool erf = false, bool prob_label = false);
	MatrixXd get_maskMatrix();
	void set_maskMatrix(MatrixXd &_mask_matrix);
	void getModel(VectorXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin, double& bias);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	VectorXd gradient(VectorXd &W);
	double logPosterior(VectorXd &W, bool precompute = true);
protected:
	bool with_mask;
 	MatrixXd mask_matrix;
 	Mask_CPU_LogisticRegression logistic_regression;
};

#endif // Mask_CPU_HAMILTONIAN_MC_H