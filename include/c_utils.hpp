#ifndef C_UTILS_H
#define C_UTILS_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace Eigen;

class C_utils
{
public:
	C_utils();
	void read_Labels(const string& filename,VectorXi& labels,int rows);
	void read_Data(const string& filename,MatrixXd& data,int rows, int cols);
	void classification_Report(VectorXi &test, VectorXi &predicted);
	void print(VectorXi &test, VectorXi &predicted);
	void classification_Report_d(VectorXi &test, VectorXd &predicted);
	int get_Rows(const string& filename);
	int get_Cols(const string& filename, char separator);
	vector<int> get_Classes(VectorXi labels);
private:
	bool initialized;
};

#endif