#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "likelihood/CPU_logistic_regression.hpp"
#include "utils/c_utils.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;


int main()
{
  C_utils utils;

  string datap_csv_path, datan_csv_path, datat_csv_path;
  MatrixXd dataPos, dataNeg, dataTest;

  datap_csv_path = "dataPos.csv";
  datan_csv_path = "dataNeg.csv";
  datat_csv_path = "dataTest.csv";
  
  cout << "Read Data" << endl;

  utils.read_Data(datap_csv_path,dataPos);
  utils.read_Data(datan_csv_path,dataNeg);
  utils.read_Data(datat_csv_path,dataTest);
  MatrixXd data(dataPos.rows()+dataNeg.rows(), dataPos.cols());
  data << dataPos ,dataNeg;
  VectorXd labels(dataPos.rows()+dataNeg.rows());
  labels << VectorXd::Ones(dataPos.rows()) , VectorXd::Zero(dataNeg.rows());

  cout << "Train" << endl;
  double lambda = 100.0;
  CPU_LogisticRegression logistic_regression;
  logistic_regression.init(data, labels, lambda, false, false, true);
  logistic_regression.train(1000,0.99,1e-1);

  cout << "Predict" << endl;
  VectorXd predicted_labels = logistic_regression.predict(dataTest, true);
  double max_prob = 0.0;
  for (int i = 0; i < predicted_labels.rows(); ++i){
      max_prob=MAX(max_prob,predicted_labels(i));
    }
  cout<< max_prob << endl;

  //cout << utils.random_generator(10)/10. << endl;
  return 0;
}