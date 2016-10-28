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
#include "../utils/utils.hpp"


using namespace Eigen;

/*** CLASS dirichlet ***/

class dirichlet{
	public:
		//constructors
		dirichlet();
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
        double log_likelihood(const Ref<const VectorXd>&counts);
		void fit_fixedPoint(MatrixXd& counts,int maxIter,double tol);
		void dirichlet_moment_match(const Ref<const MatrixXd>& proportions, const Ref<const MatrixXd>& weigths);
        void dirichlet_moment_match(const Ref<const MatrixXd>& counts);
        void fit_betabinom_minka_alternating(MatrixXd& counts, int maxiter, double tol);
	private:
		VectorXd alpha;
	    VectorXd m;
		double s;
		void polya_fit_m(MatrixXd& counts,double tol);
		void s_derivatives(MatrixXd& counts, double *g,double *h);
		double stable_a2(MatrixXd& counts);
		void polya_fit_s(MatrixXd& counts,double tol);		
};

#endif
