#include "dpp.hpp"

DPP::DPP(){
}

MatrixXd DPP::squared_exponential_kernel(MatrixXd X, double nu, double sigma_f){
    MatrixXd Cov;
    int Xnrow = X.rows();
    int Xncol = X.cols();
    Cov.resize(Xnrow,Xnrow);
    for(int i = 0; i < (Xnrow-1); ++i){      
      for(int j = i + 1; j < Xnrow; ++j){
        double S = 0;
        for(int k = 0; k < Xncol; ++k){
          double d = ( X(i,k) - X(j,k));
          S += exp(nu) * d * d;
        }
        Cov(i,j) = exp(sigma_f - 0.5 * S);
        Cov(j,i) = Cov(i,j);
      }
    }
    for(int i = 0; i < Xnrow; ++i){
        Cov(i,i) =  exp(sigma_f);
    }
    return Cov;
} 

vector<Rect> DPP::run(vector<Rect> preDetections, VectorXd &detectionWeights,VectorXd &penaltyWeights, MatrixXd &featureValues, 
	double alpha, double lambda, double beta, double mu, double epsilon)
{

	VectorXd qualityTerm = get_quality_term(detectionWeights, penaltyWeights, lambda,alpha, beta);
	MatrixXd similarityTerm = featureValues * featureValues.adjoint();

	vector<int> top = solve(qualityTerm, similarityTerm, epsilon);	
	
	vector<Rect> respDPP;
	for (size_t i = 0; i < top.size(); ++i)
	{
		respDPP.push_back(preDetections.at(top.at(i)));
	}
	
	return respDPP;
	
}

VectorXd DPP::get_quality_term(VectorXd &detectionWeights, VectorXd &penaltyWeights,double lambda, double alpha, double beta){
	VectorXd qt=lambda*detectionWeights.array()+(1-lambda)*penaltyWeights.array();
	return qt;
}

MatrixXd DPP::get_similarity_term(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu){
	/****
	 ****	Get similarity term
	 ****	S = w * S^c + (1 - w) * S^s
	 ****/

	//MatrixXd Ss = intersectionArea.array() / sqrtArea.array();
	//MatrixXd Sc = squared_exponential_kernel(featureValues, 0, 0);
	//MatrixXd Sc = featureValues * featureValues.adjoint();
	//MatrixXd S = mu * Ss.array() + (1 - mu) * Sc.array();
	//MatrixXd S = featureValues * featureValues.adjoint();
	MatrixXd S = squared_exponential_kernel(featureValues, 0, 0);
	return S;
}

vector<int> DPP::solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon){
	VectorXi remained = VectorXi::LinSpaced(qualityTerm.size(), 0, qualityTerm.size() - 1);
	int selected;
	double oldObj, prodQ;
	oldObj = qualityTerm.maxCoeff(&selected);
	
	vector<int> top;
	top.push_back(selected);
	MatrixXd oldS = MatrixXd::Identity(1,1);
	prodQ = oldObj;

	while(true){
		double maxObj_ = 0;
		copy( remained.data() + selected + 1, remained.data() + remained.size(), remained.data() + selected ); // delete selected item
		remained.conservativeResize(remained.size() - 1);

		MatrixXd newS = MatrixXd::Identity( oldS.rows() + 1, oldS.cols() + 1 );
		MatrixXd maxS( oldS.rows() + 1, oldS.cols() + 1 );
		newS.block(0,0, oldS.rows(), oldS.cols()) << oldS;

		MatrixXd S_top(top.size(), similarityTerm.cols());
		for (size_t i = 0; i < top.size(); ++i)
		{
			//S_top.row(i) << similarityTerm.row(i);
			S_top.row(i) << similarityTerm.row(top.at(i));
		}

		for (int i = 0; i < remained.size(); ++i)
		{
			VectorXd tmp = S_top.col(remained(i));

			newS.block(0, newS.cols() - 1, newS.rows() - 1, 1) << tmp;
			newS.block(newS.rows() - 1, 0, 1, newS.cols() - 1) << tmp.transpose();

			double obj_ = qualityTerm(remained(i)) * newS.determinant();
			//cout << obj_  << ","<< maxObj_ << endl;
			if (obj_ > maxObj_)
			{
				selected = i;
				maxObj_ = obj_;
				maxS = newS;
			}

		}

		double maxObj = prodQ * maxObj_ ;
		cout << maxObj / oldObj << ","<<  1+epsilon << endl;
		if ( (maxObj / oldObj) > (epsilon) )
		{
			top.push_back(remained(selected));
			oldObj = maxObj;
			oldS = maxS;
			prodQ = prodQ * qualityTerm(remained(selected));
		}
		else{
			break;
		}
	}

	return top;
}




