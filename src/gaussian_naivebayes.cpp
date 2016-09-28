#include "../include/gaussian_naivebayes.hpp"

GaussianNaiveBayes::GaussianNaiveBayes(){}

GaussianNaiveBayes::GaussianNaiveBayes(Mat& _positiveFeatureValue, Mat& _negativeFeatureValue){
	sampleFeatureValue = &_positiveFeatureValue;
	negativeFeatureValue = &_negativeFeatureValue;
	theta_y_mu = VectorXd(sampleFeatureValue->rows);
	theta_y_sigma = VectorXd(sampleFeatureValue->rows);
	initialized = true;
}

void GaussianNaiveBayes::fit(){
	positive_likelihood.clear();
	negative_likelihood.clear();
	Scalar muTemp, sigmaTemp;
	for (int i = 0; i < sampleFeatureValue->rows; i++)
	{
		meanStdDev(sampleFeatureValue->row(i), muTemp, sigmaTemp);
		theta_y_mu[i] = muTemp.val[0];
		theta_y_sigma[i] = sigmaTemp.val[0];
		Gaussian haar_feature(theta_y_mu[i], theta_y_sigma[i]);
		positive_likelihood.push_back(haar_feature);
	}

	for (int i = 0; i < negativeFeatureValue->rows; i++){
        meanStdDev(negativeFeatureValue->row(i), muTemp, sigmaTemp);
        Gaussian haar_feature((double)muTemp.val[0],(double)sigmaTemp.val[0]);
        negative_likelihood.push_back(haar_feature);
    }
}

void GaussianNaiveBayes::setSampleFeatureValue(Mat& _sampleFeatureValue){
	sampleFeatureValue = &_sampleFeatureValue;
}

float GaussianNaiveBayes::test(int index_particle){
	float prob_haar=0.0f;
	for(int j=0;j<sampleFeatureValue->rows;j++){
 		//cout << haar.featureNum << "," << i << "," << j << endl; 
		float haar_prob=sampleFeatureValue->at<float>(j,index_particle);
        prob_haar += positive_likelihood.at(j).log_likelihood(haar_prob)-
        			negative_likelihood.at(j).log_likelihood(haar_prob);
    }
    return prob_haar;
}