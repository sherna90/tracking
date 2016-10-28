#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include "../include/adaboost.hpp"
#include "../include/c_utils.hpp"
#include <string>
#include <fstream>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
  string data_csv_path, labels_csv_path, algorithm;
  int n_estimators;
  int rows, cols, train_partition;
  double alpha, learning_rate;

  if(argc == 8){
    data_csv_path = argv[1];
    labels_csv_path = argv[2];
    n_estimators= atoi(argv[3]);
    algorithm = argv[4];
    alpha = atof(argv[5]);
    learning_rate = atof(argv[6]);
    train_partition = atoi(argv[7]);
  }
  else{
    cout << "Arguments :" << argc-1 << "/7" << endl;
    cout << "\nRun: ./test_adaboost [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "\t<string> path of data csv file" << endl;
    cout << "\t<string> path of labels csv file" << endl;
    cout << "\t<int> Number of estimators" << endl;
    cout << "\t<string> Adaboost algorithm: samme or samme.r" << endl;
    cout << "\t<double> alpha - Multinomial algorithm, any for Gaussian" << endl;
    cout << "\t<double> learning rate" << endl;
    cout << "\t<int> Number of rows data train\n" << endl;
    
    exit(1);
  }
  
  MatrixXd data;
  VectorXi labels;

  C_utils utils;

  rows = utils.get_Rows(labels_csv_path);
  cols = utils.get_Cols(data_csv_path, ',');
  utils.read_Data(data_csv_path,data,rows,cols);
  utils.read_Labels(labels_csv_path,labels,rows);

  MatrixXd data_train =  data.block(0,0 ,train_partition, data.cols());
  MatrixXd data_test =  data.block(train_partition,0 ,data.rows()-train_partition, data.cols());
  VectorXi labels_train =  labels.head(train_partition);
  VectorXi labels_test =  labels.tail(labels.rows()-train_partition);

  VectorXi predicted_labels;

	Adaboost ensemble(algorithm, n_estimators, alpha, learning_rate);
	ensemble.fit(data_train, labels_train);
  predicted_labels = ensemble.predict(data_test);

  utils.classification_Report(labels_test, predicted_labels);
  //utils.print(labels_test, predicted_labels);


  return 0;
}

