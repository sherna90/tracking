/**
 * @file dirichlet.h
 * @brief dirichlet density 
 * @author Guillermo Becerra
 */
#ifndef DIRICHLET
#define DIRICHLET

#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "utils.hpp"


using namespace Eigen;

/*** CLASS dirichlet ***/

class dirichlet{
	public:
		//constructors
		dirichlet(VectorXd a);
		dirichlet(int n);
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
        void dirichlet_moment_match(MatrixXd counts);
	private:
		VectorXd alpha;
	    VectorXd m;
		double s;
		
};

#endif