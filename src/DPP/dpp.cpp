#include "dpp.hpp"

DPP::DPP(){}

vector<Rect> DPP::run(vector<Rect> preDetections, VectorXd &detectionWeights, MatrixXd &featureValues, double alpha, double lambda, double beta, double mu, double epsilon)
{
	VectorXd area(preDetections.size());
	cout << "preDetections size: " << preDetections.size() << endl;
	MatrixXd intersectionArea(preDetections.size(), preDetections.size());

	for (size_t i = 0; i < preDetections.size(); ++i)
	{
		Rect bbox = preDetections.at(i);
		area(i) = bbox.width * bbox.height;
		//cout << "bbox.width: " << bbox.width << "\tbbox.height: " << bbox.height << "\tarea: " << area(i) << endl;
		
		for (size_t j = 0; j < preDetections.size(); ++j)
		{	Rect bbox2 = preDetections.at(j);
			intersectionArea(i,j) = double((bbox & bbox2).area());
		}
			
	}

	MatrixXd sqrtArea = area.cwiseSqrt() * area.cwiseSqrt().adjoint();
	MatrixXd rIntersectionArea = intersectionArea.array() / area.replicate(1, area.size()).adjoint().array();

	VectorXd nContain = VectorXd::Zero(rIntersectionArea.rows());
	for (int i = 0; i < rIntersectionArea.rows(); ++i)
	{
		for (int j = 0; j < rIntersectionArea.cols(); ++j)
		{
			if(rIntersectionArea(i,j) == 1)
				nContain(i) += 1;
		}
	}


	nContain = nContain.array() - 1;
	VectorXd nPenalty = nContain.array().exp().pow(lambda);
	
	//cout << "qt" << endl;
	VectorXd qualityTerm = getQualityTerm(detectionWeights, nPenalty, alpha, beta);
	//cout << qualityTerm << endl;
	
	//cout << "st" << endl;
	MatrixXd similarityTerm = getSimilarityTerm(featureValues, intersectionArea, sqrtArea, mu);
	//cout << similarityTerm.col(0) << endl;

	//cout << "solve" << endl;
	vector<int> top = solve(qualityTerm, similarityTerm, epsilon);	

	vector<Rect> respDPP;
	for (size_t i = 0; i < top.size(); ++i)
	{
		respDPP.push_back(preDetections.at(top.at(i)));
	}
	
	return respDPP;
	
}

VectorXd DPP::getQualityTerm(VectorXd &detectionWeights, VectorXd &nPenalty, double alpha, double beta){
	/*** 
	 ***	Get quality term 
	 ***	q = alpha * log(1 + s) + beta
	 ***/

	VectorXd qt = detectionWeights.cwiseProduct(nPenalty);
	//double maxQt = qt.maxCoeff();
	qt = qt.array() / qt.maxCoeff();
	qt = qt.array() + 1;
	qt = qt.array().log() / log(10);
	qt = alpha * qt.array() + beta;
	qt = qt.array().square();
	return qt;
}

MatrixXd DPP::getSimilarityTerm(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu){
	/****
	 ****	Get similarity term
	 ****	S = w * S^c + (1 - w) * S^s
	 ****/

	MatrixXd Ss = intersectionArea.array() / sqrtArea.array();
	MatrixXd Sc = featureValues * featureValues.adjoint();
	MatrixXd S = mu * Ss.array() + (1 - mu) * Sc.array();
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

			/*cout << "newS size: " << newS.rows() << "," << newS.cols() << endl;
			cout << "newS block size: " << newS.block(0, newS.cols() - 1, newS.rows() - 1, 1).rows() << "," << newS.block(0, newS.cols() - 1, newS.rows() - 1, 1).cols() << endl;
			cout << "newS block size: " << newS.block(newS.rows() - 1, 0, 1, newS.cols() - 1).rows() << "," << newS.block(newS.rows() - 1, 0, 1, newS.cols() - 1).cols() << endl;
			cout << "tmp size: " << tmp.size() << endl;
			cout << "tmp rows: " << tmp.rows() << "\tcols: " << tmp.cols() << endl;*/

			newS.block(0, newS.cols() - 1, newS.rows() - 1, 1) << tmp;
			newS.block(newS.rows() - 1, 0, 1, newS.cols() - 1) << tmp.transpose();

			double obj_ = qualityTerm(remained(i)) * newS.determinant();
			
			if (obj_ > maxObj_)
			{
				selected = i;
				maxObj_ = obj_;
				maxS = newS;
			}

		}

		double maxObj = prodQ * maxObj_ ;

		if ( (maxObj / oldObj) > (1 + epsilon) )
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




