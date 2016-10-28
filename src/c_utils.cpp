#include <iostream>
#include "../include/c_utils.hpp"

typedef std::numeric_limits< double > dbl;

C_utils::C_utils(){
  initialized=true;
}

void C_utils::read_Labels(const string& filename,VectorXi& labels,int rows) {
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    string line;
    int row = 0;
    labels.resize(rows);
    while (getline(file, line)) {
      if (row < rows){
        int item=atoi(line.c_str());
        labels(row)=item;
        row++;
      }
    }
    file.close();
}


void C_utils::read_Data(const string& filename,MatrixXd& data,int rows, int cols) {
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    string line, cell;
    int row = 0,col;
    data.resize(rows, cols);
    while (getline(file, line)) {
      col=0;
      stringstream csv_line(line);
      if (row < rows){
        while (getline(csv_line, cell, ',')){
          if (col<cols){
            double item=atof(cell.c_str());
            data(row,col)=item;
            col++;
          }
        }
      row++;
      }
    }
    file.close();
}

void C_utils::classification_Report(VectorXi &test, VectorXi &predicted)
{
  int count=0;
  for (int i=0; i < test.size(); i++) if (int(predicted(i)) == test(i)) count++;
  double percent = double(count) / double(test.size());
  cout << setprecision(10) << fixed;
  cout << percent*100 << endl;
  
}

void C_utils::classification_Report_d(VectorXi &test, VectorXd &predicted)
{
  int count=0;
  for (int i=0; i < test.size(); i++) if (int(predicted(i)) == test(i)) count++;
  double percent = double(count) / double(test.size());
  cout << setprecision(10) << fixed;
  cout << percent*100 << endl;
  
}

void C_utils::print(VectorXi &test, VectorXi &predicted)
{
  for (int i=0; i < test.size(); i++) cout << test(i) << " " << predicted(i) << endl;
}

int C_utils::get_Rows(const string& filename) {
  int number_of_lines = 0;
  string line;
  ifstream file(filename.c_str());
  while (getline(file, line)) ++number_of_lines;
  return number_of_lines;
}

int C_utils::get_Cols(const string& filename, char separator) {
  int number_of_lines = 0;
  string line, token;
  ifstream file(filename.c_str());
  getline(file, line);
  istringstream aux(line);
  while (getline(aux, token, separator)) ++number_of_lines;
  return number_of_lines;
}

vector<int> C_utils::get_Classes(VectorXi labels){
  vector<int> classes;
  classes.push_back(labels(0));
  for (int i = 1; i < labels.rows(); ++i) {
    int count = 0;
    for (unsigned int k = 0; k < classes.size(); ++k) {
      if (labels(i)!= int(classes.at(k))) count++;
    }
    if (count == int(classes.size())) classes.push_back(labels(i));
  }
  sort(classes.begin(), classes.end());
  return classes;
}
