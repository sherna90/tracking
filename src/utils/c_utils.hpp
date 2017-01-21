// Author: Diego Vergara
#ifndef C_UTILS_H
#define C_UTILS_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <set>

using namespace std;
using namespace Eigen;

typedef struct{
  double precision;
  double accuracy;
  double recall;
  double f1score;
  double support;
}Metrics;

class C_utils
{
public:
	C_utils();
	void read_Labels(const string& filename,VectorXi& labels,int rows);
	void read_Data(const string& filename,MatrixXd& data,int rows, int cols);
	void print(VectorXi &test, VectorXi &predicted);
	void classification_Report_d(VectorXi &test, VectorXd &predicted);
	void classification_Report(VectorXd &test, VectorXi &predicted);
	void classification_Report(VectorXi &test, VectorXi &predicted);
	int get_Rows(const string& filename);
	int get_Cols(const string& filename, char separator);
	vector<int> get_Classes(VectorXi labels);
	vector<int> get_Classes_d(VectorXd labels);
	
	map<pair<int,int>, int> confusion_matrix(VectorXi &test, VectorXi &predicted, bool print=true);
	map<pair<int,int>, int> confusion_matrix(VectorXi &test, VectorXd &predicted, bool print=true);
	map<pair<int,int>, int> confusion_matrix(VectorXd &test, VectorXd &predicted, bool print=true);
	
	map<int, double> precision_score(VectorXi &test, VectorXi &predicted, bool print=true);
	map<int, double> precision_score(VectorXi &test, VectorXd &predicted, bool print=true);
	map<int, double> precision_score(VectorXd &test, VectorXd &predicted, bool print=true);
	map<int, double> precision_score(map<pair<int,int>, int> confusion_matrix, bool print=false);

	map<int, double> accuracy_score(VectorXi &test, VectorXi &predicted, bool print=true);
	map<int, double> accuracy_score(VectorXi &test, VectorXd &predicted, bool print=true);
	map<int, double> accuracy_score(VectorXd &test, VectorXd &predicted, bool print=true);
	map<int, double> accuracy_score(map<pair<int,int>, int> confusionMatrix, bool print=false);
	
	map<int, double> recall_score(VectorXi &test, VectorXi &predicted, bool print=true);
	map<int, double> recall_score(VectorXi &test, VectorXd &predicted, bool print=true);
	map<int, double> recall_score(VectorXd &test, VectorXd &predicted, bool print=true);
	map<int, double> recall_score(map<pair<int,int>, int> confusionMatrix, bool print=false);

	map<int, double> f1_score(VectorXi &test, VectorXi &predicted, bool print=true);
	map<int, double> f1_score(VectorXi &test, VectorXd &predicted, bool print=true);
	map<int, double> f1_score(VectorXd &test, VectorXd &predicted, bool print=true);
	map<int, double> f1_score(map<pair<int, int>, int> confusionMatrix, bool print=false);

	map<int, double> support_score(VectorXi &test);
	map<int, double> support_score(VectorXd &test);
	map<int, double> support_score(map<pair<int, int>, int> confusionMatrix);

	map<int, Metrics> report(VectorXi &test, VectorXi &predicted, bool print=true);
	map<int, Metrics> report(VectorXi &test, VectorXd &predicted, bool print=true);
	map<int, Metrics> report(VectorXd &test, VectorXd &predicted, bool print=true);
	map<int, Metrics> report(map<pair<int, int>, int> confusionMatrix, bool print=true);
private:
	bool initialized;
};

#endif