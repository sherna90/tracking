/**
 * @file dirichlet.cpp
 * @brief polya density 
 * @author Guillermo Becerra
 */
#ifndef DIRICHLET
#define DIRICHLET

#include <stdlib.h>
#include <cmath>
#include <Eigen/Dense>
#include "utils.h"


using namespace std;
using namespace Eigen;

/*** CLASS POYLA ***/

class Polya{
	public:
		//constructors
		Polya(VectorXd a);
		Polya(int n);
		//getters
		VectorXd getAlpha(){return alpha;};
		VectorXd getM(){return m;};
		double getS(){return s;};
		//setters
		void setAlpha(VectorXd a);
		//methods
		void meanprecision();
        double log_likelihood(VectorXd counts);
		void fit_fixedPoint(MatrixXd counts,int maxIter,double tol);
		void dirichlet_moment_match(MatrixXd proportions, MatrixXd weigths);
        void polya_moment_match(MatrixXd counts);
	private:
		VectorXd alpha;
	    VectorXd m;
		double s;
		
};

#endif