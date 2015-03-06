/**
 * @file dirichlet.cpp
 * @brief polya density 
 * @author Guillermo Becerra
 */
#include <stdlib.h>
#include <cmath>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

namespace Util {

	double lnchoose(int  n, int m){
		
		double nf=lgamma(n+1);
		double mf=lgamma(m+1);
		double nmmnf = lgamma(n-m+1);
		return (nf-(mf+nmmnf));
	
	}

	double bhattarchaya(VectorXd m1, VectorXd m2){
		
		RowVectorXd coef;
		coef=m1.cwiseProduct(m2).cwiseSqrt()/sqrt(m1.sum()*m2.sum());
  		return sqrt(1-coef.sum());
    }

	void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
	{
	
	    unsigned int numRows = matrix.rows()-1;
	    unsigned int numCols = matrix.cols();

	    if( rowToRemove < numRows )
	        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

	    matrix.conservativeResize(numRows,numCols);
	    
	}

	void flat(MatrixXd& mat){
	
		mat.resize(mat.rows()*mat.cols(),1);
		
	}
	VectorXd average(MatrixXd a,MatrixXd weigths, int axis){
		MatrixXd r = a;
		Util::flat(weigths);
 		
		if(axis==0){

			for(int i=0;i<r.cols();i++){
				r.col(i)= r.col(i).array()*weigths.array();	
			}
			return (1/weigths.sum())*r.colwise().sum();	
		
		}else if(axis==1){

			for(int i=0;i<r.rows();i++){
				r.row(i)= r.row(i).array()*weigths.transpose().array();	
			}
			return (1/weigths.sum())*r.rowwise().sum().transpose();
		
		}else{
			cout << "invalid argument on average function AVERAGE" << endl;
			exit(EXIT_FAILURE);
		}
		
	}
	double median(MatrixXd med){
		MatrixXd aux = med;
		flat(aux);
		int n = aux.rows();
		if(n%2==0){
			return (aux(n/2,0)+aux(n/2-1,0))/2.0;

		}else{
			return aux(n/2,0);
		}
	}
	// Utils for digamma from http://fastapprox.googlecode.com/svn/trunk/fastapprox/src/fastonebigheader.h
	
	static inline float fastlog2 (float x)
	{
	  union { float f; uint32_t i; } vx = { x };
	  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
	  float y = vx.i;
	  y *= 1.1920928955078125e-7f;

	  return y - 124.22551499f
	           - 1.498030302f * mx.f 
	           - 1.72587999f / (0.3520887068f + mx.f);
	}

	static inline float fastlog (float x)
	{
	  return 0.69314718f * fastlog2 (x);
	}


	static inline float fastdigamma (float x)
	{
	  float twopx = 2.0f + x;
	  float logterm = fastlog (twopx);

	  return (-48.0f + x * (-157.0f + x * (-127.0f - 30.0f * x))) /
	         (12.0f * x * (1.0f + x) * twopx * twopx)
	         + logterm;
	}

	MatrixXd psi(MatrixXd mat){
		MatrixXd res(mat.rows(),mat.cols());

		for(int i=0;i<mat.rows();i++){
			for(int j=0;j<mat.cols();j++){
				res(i,j)=fastdigamma(mat(i,j));
			}
		}
		return res;
	}
	float psi(float x){
		
		return fastdigamma(x);
	}

}

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
Polya::Polya(int n){
	 alpha=VectorXd::Ones(n);
	 meanprecision();
}
Polya::Polya(VectorXd a)
{
    setAlpha(a);
    meanprecision();
}


void Polya::setAlpha(VectorXd a){
	alpha=a;
	meanprecision();
}

double Polya::log_likelihood(VectorXd counts){
    VectorXd log_counts=VectorXd::Zero(counts.size());
    VectorXd log_alpha=VectorXd::Zero(alpha.size());
    VectorXd log_alpha_counts=VectorXd::Zero(alpha.size());
    double csum=counts.sum();
    for(int i=0;i<counts.size();i++){
        log_counts[i]=lgamma(counts[i]+1);
        log_alpha[i]=lgamma(alpha[i]);
        log_alpha_counts[i]=lgamma(alpha[i]+counts[i]);
    }
    double loglike=0.0;
    loglike+=lgamma(csum+1)-log_counts.sum();
    loglike+=lgamma(s) - lgamma(s + csum);
    loglike+=(log_alpha_counts-log_alpha).sum();
    return loglike;
}

void Polya::meanprecision(){
	s= alpha.sum();
	m= (1.0f/s)*alpha;
}

void Polya::dirichlet_moment_match(MatrixXd proportions, MatrixXd weigths){
	VectorXd a;
	VectorXd m2;
	VectorXd aok,m2ok;
	double res=0;
	a= Util::average(proportions,weigths,0);
	m2 = Util::average(proportions.cwiseProduct(proportions),weigths,0);
	
	aok = VectorXd(a.size());
	m2ok = VectorXd(m2.size());
	
	int k=0;
	for(int i=0;i<a.size();i++){
		if(a[i]>0){
			aok[k]=a[i];
			m2ok[k]=m2[i];
			k++;
		}else{
			aok.conservativeResize(aok.size()-1);
			m2ok.conservativeResize(m2ok.size()-1);
		}
	}

	res=Util::median((aok - m2ok).cwiseQuotient((m2ok - aok.cwiseProduct(aok))));

	alpha=a*res;
	
}

void Polya::polya_moment_match(MatrixXd counts){
	MatrixXd norm_sum = counts.rowwise().sum();
	
	for(int i=0;i<counts.rows();i++){
		if(norm_sum(i,0)!=0){
			counts.row(i)= counts.row(i)*(1/norm_sum(i,0)); //	
		}else{
			continue;
		}
	}
	dirichlet_moment_match(counts,norm_sum);
}

void Polya::fit_fixedPoint(MatrixXd counts,int maxIter,double tol){ //incomplete
	int train=counts.rows();
	int D=counts.cols();
	int iter=0;
	VectorXd old_alp ;
	MatrixXd c;
	VectorXd d;
	double change = 2*tol;
    //counts = counts[sum(counts.A, axis=1) > 0, :]
    MatrixXd auxCounts = counts.rowwise().sum();
	for(int i=0;i<auxCounts.rows();i++){
		 if(auxCounts(i,0)<0){
		 	Util::removeRow(auxCounts,i);
		 	Util::removeRow(counts,i);
		 	i--;
		 }
	}

	//alpha = array(polya_moment_match(counts)).flatten()
 	polya_moment_match(counts);
	c = MatrixXd::Zero(train,D);
	d = VectorXd::Zero(train);

	while(change > tol && iter < maxIter)
	{
		old_alp=alpha;
		for (int i=0;i<train;i++){
			c.row(i)=Util::psi(counts.row(i)+alpha.transpose())-Util::psi(alpha.transpose());
			d[i]= Util::psi(counts.row(i).sum()+alpha.sum())-Util::psi(alpha.sum());
		}
		auxCounts=(c.colwise().sum())*(1.0/d.sum());
		alpha=alpha.cwiseProduct(auxCounts.transpose());
		change = (alpha-old_alp).cwiseAbs().maxCoeff();
		iter++;
	}
	
}