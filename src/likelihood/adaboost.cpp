// Author: Diego Vergara
#include "adaboost.hpp"
#include "../utils/c_utils.hpp"

Adaboost::Adaboost()
{
    initialized=false;
}

Adaboost::Adaboost(string _algorithm, int _n_estimators, double _alpha, double _learning_rate)
{
    algorithm = _algorithm;
    n_estimators = _n_estimators;
    M_alpha = _alpha;
    learning_rate = _learning_rate;
    initialized=true;
}


double Adaboost::boost_discrete(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers){
  

  VectorXi predicted_labels;
  VectorXd index = VectorXd::Zero(n_data);
  ///////////////////////////////
  //MultinomialNaiveBayes naive_bayes(*getdata(),*getlabels());
  //naive_bayes.fit(M_alpha, w);
  //predicted_labels=naive_bayes.test(*getdata());

  GaussianNaiveBayes naive_bayes(*getdata(),*getlabels());
  naive_bayes.fit(w);
  predicted_labels=naive_bayes.test(*getdata());

  ///////////////////////////////
  classifiers.push_back(naive_bayes);
  for (int i = 0; i < predicted_labels.rows(); ++i) index(i) = ((predicted_labels(i) == (*getlabels())(i)) ? 0: 1);
  double e = (w.cwiseProduct(index)).colwise().sum()*(w.colwise().sum()).inverse();
  if (e <= 0.0){
    cout << "e negativo" << endl;
    w = VectorXd::Zero(n_data);
    errors(iteration) = 0.0;
    return -1.0;
  }  

  if (e >= 1.0 - (1.0 / n_classes)){
    if (iteration == 0) cout << "Critico: Error en primera Iteracion" << endl;
    cout << "e muy grande" << endl;
    w = VectorXd::Zero(n_data);
    errors(iteration) = 0.0;
    return 0.0;
  }
  double alpha = learning_rate * (log((1.0-e) / e) + log(n_classes-1.0)); 
  for (int i = 0; i < n_data; ++i) index(i) *= (((w(i) > 0) or (alpha < 0)) ? 1: 0); //
  if (iteration != (n_estimators-1)) w = w.array() * ((index*alpha).array().exp());

  
  errors(iteration) = e;
  
  return alpha;
}

double Adaboost::boost_real(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers){
  

  VectorXi predicted_labels(n_data);
  VectorXd index(n_data);
  VectorXd alpha(n_data);
  MatrixXd proba;
  
  ///////////////////////////////
  //MultinomialNaiveBayes naive_bayes(*getdata(),*getlabels());
  //naive_bayes.fit(M_alpha, w);
  //proba=naive_bayes.get_proba(*getdata());

  GaussianNaiveBayes naive_bayes(*getdata(),*getlabels());
  naive_bayes.fit(w);
  proba = naive_bayes.get_proba(*getdata());

  MatrixXf::Index   maxIndex[proba.rows()];
  for (int j =0; j< proba.rows(); ++j){
    proba.row(j).maxCoeff(&maxIndex[j]);
    predicted_labels(j) = maxIndex[j];
   }
  //predicted_labels=naive_bayes.test(*getdata());
  ///////////////////////////////
  classifiers.push_back(naive_bayes);

  for (int i = 0; i < predicted_labels.rows(); ++i) index(i) = ((predicted_labels(i) == (*getlabels())(i)) ? 0: 1);
  double e = (w.cwiseProduct(index)).colwise().sum()*(w.colwise().sum()).inverse();

  if (e <= 0.0){
    cout << "e negativo" << endl;
    w = VectorXd::Zero(n_data);
    errors(iteration) = 0.0;
    return -1.0;
  }  

  MatrixXd y_coding = MatrixXd::Ones(n_data, n_classes)*(-1. / (n_classes -1.));
  for (int i = 0; i < n_data; ++i) y_coding(i ,(*getlabels())(i)) = 1.0; // ojo indices
  
  alpha = (-1. * learning_rate * (((n_classes -1.) / n_classes) * ((y_coding.array().cwiseProduct(proba.array())).rowwise().sum())));

  for (int i = 0; i < n_data; ++i) index(i) = (((w(i) > 0) or (alpha(i) < 0)) ? 1: 0);
  if (iteration != (n_estimators-1)) w = w.array() * ((alpha.array()*index.array()).exp());
  

  
  errors(iteration) = e;
  
  return 1.0;
}

void Adaboost::fit(MatrixXd &data, VectorXi &labels){
  dX=&data;
  lX=&labels;
  C_utils utils;
  classes  = utils.get_Classes(*(&labels));
  n_classes = classes.size();
  dim = getdata()->cols();
  n_data = getdata()->rows();

  if (initialized){

    VectorXd w = VectorXd::Ones(n_data)/ n_data;
    alphas = VectorXd::Zero(n_estimators);
    VectorXd errors = VectorXd::Ones(n_estimators);

    for (int i = 0; i < n_estimators; ++i){
      cout << i << endl;

      if (algorithm == "samme.r"){
        alphas(i) = boost_real(w, i, errors, classifiers);  
      }
      else{
        alphas(i) = boost_discrete(w, i, errors, classifiers);  
      }   
      
      double w_sum = w.sum();

      if ((w_sum <= 0) or ( errors(i) == 0.0)){
        cout << "Break" << endl;
        n_estimators = i+1;
        break;
      }

      if (i < (n_estimators-1)) w /= w_sum; 
    }
  }
}

MatrixXd Adaboost::get_proba(GaussianNaiveBayes classifier){
  MatrixXd proba = classifier.get_proba(*gettest());
  VectorXd temp(proba.rows());
  temp = (1. / n_classes) * proba.rowwise().sum();
  for (int i = 0; i < proba.rows(); ++i) for (int j = 0; j < proba.cols(); ++j) proba(i,j) -= temp(i);
  return (n_classes - 1) * proba;
}

VectorXi Adaboost::predict( MatrixXd &test){
  dY=&test;
  n_data_test = gettest()->rows();
  MatrixXd pred = MatrixXd::Zero(n_data_test,n_classes);
  VectorXi result = VectorXi::Zero(n_data_test);

  if (algorithm == "samme.r"){
    for (int i = 0; i < n_estimators; ++i) pred += get_proba(classifiers.at(i));
  }
  else{
    VectorXi temp_Yhat;
    for (int i = 0; i < n_estimators; ++i){
      temp_Yhat= classifiers.at(i).test(*gettest());
      for (int j =0; j< n_classes; ++j) for (int k = 0; k < temp_Yhat.rows(); ++k) pred(k,j) += ((temp_Yhat(k) != classes.at(j)) ? 1: 0)*alphas(i);
    }
  }
  pred /= alphas.sum();
  if (n_classes == 2){
    pred.col(0) *= -1;
    pred.col(0) += pred.col(1);
    for (int j =0; j< pred.rows(); ++j) result(j) = ((pred(j,0) > 0) ? classes.at(1): classes.at(0));
  }
  else{
    MatrixXf::Index   maxIndex[pred.rows()];
    for (int j =0; j< pred.rows(); ++j){
      pred.row(j).maxCoeff(&maxIndex[j]);
      result(j) = maxIndex[j];
     }
  }
  //C_utils utils;
  //utils.classification_Report(*gettestlabels(), result);
  return result;
}


 MatrixXd *Adaboost::getdata() 
{
    return dX;
}

 MatrixXd *Adaboost::gettest() 
{
    return dY;
}

 VectorXi *Adaboost::getlabels() 
{
    return lX;
}
