// Author: Diego Vergara
#include <iostream>
#include "c_utils.hpp"

typedef std::numeric_limits< double > dbl;

C_utils::C_utils(){
  initialized=true;
}

void C_utils::read_Labels(const string& filename, VectorXi& labels, int rows) {
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

void C_utils::read_Data(const string& filename, MatrixXd& data, int rows, int cols) {
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

vector<int> C_utils::get_Classes_d(VectorXd labels){
  vector<int> classes;
  classes.push_back(round(labels(0)));
  for (int i = 1; i < labels.rows(); ++i) {
    int count = 0;
    for (unsigned int k = 0; k < classes.size(); ++k) {
      if (round(labels(i))!= int(classes.at(k))) count++;
    }
    if (count == int(classes.size())) classes.push_back(round(labels(i)));
  }
  sort(classes.begin(), classes.end());
  return classes;
}

map<pair<int,int>, int> C_utils::confusion_matrix(VectorXi &test, VectorXi &predicted, bool print){
  vector<int> testedClasses = get_Classes(test);
  vector<int> predictedClasses = get_Classes(predicted);
  set<int> classes;

  classes.insert(testedClasses.begin(), testedClasses.end());
  classes.insert(predictedClasses.begin(), predictedClasses.end());

  map<pair<int,int>, int> confusionMatrix;
  
  for (set<int>::iterator i = classes.begin(); i != classes.end(); ++i)
  {
    for (set<int>::iterator j = classes.begin(); j != classes.end(); ++j)
    {
      confusionMatrix[pair<int,int>(*i,*j)] = 0;
    }
  }

  for(int i = 0; i < test.size(); i++){
    confusionMatrix[pair<int,int>(test(i),predicted(i))] +=1;
  }

  if(print){
    cout << "Confusion Matrix:" << endl;
    for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
    {
      cout << "class " << it->first.first << " was predicted " << it->second << " times as class " << it->first.second << endl;
    }
  }

  return confusionMatrix;
}

map<pair<int,int>, int> C_utils::confusion_matrix(VectorXi &test, VectorXd &predicted, bool print){
  vector<int> testedClasses = get_Classes(test);
  vector<int> predictedClasses = get_Classes_d(predicted);
  set<int> classes;

  classes.insert(testedClasses.begin(), testedClasses.end());
  classes.insert(predictedClasses.begin(), predictedClasses.end());

  map<pair<int,int>, int> confusionMatrix;
  
  for (set<int>::iterator i = classes.begin(); i != classes.end(); ++i)
  {
    for (set<int>::iterator j = classes.begin(); j != classes.end(); ++j)
    {
      confusionMatrix[pair<int,int>(*i,*j)] = 0;
    }
  }
  
  for(int i = 0; i < test.size(); i++){
    confusionMatrix[pair<int,int>(test(i),predicted(i))] +=1;
  }
  
  if(print){
    cout << "Confusion Matrix:" << endl;
    for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
    {
      cout << "class " << it->first.first << " was predicted " << it->second << " times as class " << it->first.second << endl;
    }
  }

  return confusionMatrix;
}

map<int,double> C_utils::precision_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return precision_score(confusionMatrix, print);
}

map<int, double> C_utils::precision_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return precision_score(confusionMatrix, print);
}

map<int,double> C_utils::precision_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_positive;
  map<int, double> precision;

  for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_positive.find(it->first.second) == false_positive.end()){
        false_positive[it->first.second] = it->second;
      }
      else{
       false_positive[it->first.second] += it->second; 
      }
    }
  }

  for (std::map<int,double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    precision[it->first] = true_positive[it->first] / (true_positive[it->first] + false_positive[it->first]);
    if(print){
      cout << "class " << it->first << ": " << precision[it->first] << endl;
    }
  }
  
  return precision;
}

map<int, double> C_utils::accuracy_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return accuracy_score(confusionMatrix, print);
}

map<int, double> C_utils::accuracy_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return accuracy_score(confusionMatrix, print);
}

map<int, double> C_utils::accuracy_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_positive;
  map<int, double> false_negative;
  map<int, double> accuracy;

  for (std::map<pair<int,int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_positive.find(it->first.second) == false_positive.end()){
        false_positive[it->first.second] = it->second;
      }
      else{
       false_positive[it->first.second] += it->second; 
      }

      if(false_negative.find(it->first.first) == false_negative.end()){
        false_negative[it->first.first] = it->second;
      }
      else{
        false_negative[it->first.second] += it->second;
      }
    }
  }

  for (std::map<int, double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    accuracy[it->first] = true_positive[it->first] / (true_positive[it->first] + false_positive[it->first] + false_negative[it->first]);
    if(print){
      cout << "class " << it->first << ": " << accuracy[it->first] << endl;
    }
  }
  return accuracy;
}

map<int, double> C_utils::recall_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return recall_score(confusionMatrix, print);
}

map<int, double> C_utils::recall_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return recall_score(confusionMatrix, print);
}

map<int, double> C_utils::recall_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_negative;
  map<int, double> recall;

  for (std::map<pair<int,int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_negative.find(it->first.first) == false_negative.end()){
        false_negative[it->first.first] = it->second;
      }
      else{
        false_negative[it->first.second] += it->second;
      }
    }
  }

  for (std::map<int, double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    recall[it->first] = true_positive[it->first] / (true_positive[it->first] + false_negative[it->first]);
    if(print){
      cout << "class " << it->first << ": " << recall[it->first] << endl;
    }
  }
  return recall;
}

map<int, double> C_utils::f1_score(map<pair<int, int>, int> confusionMatrix, bool print){
  map<int, double> precision = precision_score(confusionMatrix);
  map<int, double> recall = recall_score(confusionMatrix);
  map<int, double> f1score;
  for (std::map<int, double>::iterator it = precision.begin(); it != precision.end(); ++it)
  {
    f1score[it->first] = 2 * (precision[it->first] * recall[it->first]) / (precision[it->first] + recall[it->first]);
    if(print){
      cout << "class " << it->first << ": " << f1score[it->first] << endl;
    }
  }
  return f1score;
}

map<int, double> C_utils::f1_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return f1_score(confusionMatrix);
}

map<int, double> C_utils::f1_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return f1_score(confusionMatrix);
}

map<int, double> C_utils::support_score(VectorXi &test){
  map<int, double> support;
  for (int i = 0; i < test.size(); ++i)
  {
    if(support.find(test(i)) == support.end()){
      support[test(i)] = 1;
    }
    else{
      support[test(i)] +=1;
    }
  }
  return support;
}

map<int, double> C_utils::support_score(VectorXd &test){
  map<int, double> support;
  for (int i = 0; i < test.size(); ++i)
  {
    if(support.find(round(test(i))) == support.end()){
      support[round(test(i))] = 1;
    }
    else{
      support[round(test(i))] +=1;
    }
  }
  return support;
}

map<int, double> C_utils::support_score(map<pair<int, int>, int> confusionMatrix){
  map<int, double> support;
  for (std::map<pair<int, int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(support.find(it->first.first) == support.end()){
      support[it->first.first] = it->second;
    }
    else{
      support[it->first.first] += it->second;
    }
  }
  return support;
}

map<int, Metrics> C_utils::report(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return report(confusionMatrix, print);
}

map<int, Metrics> C_utils::report(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return report(confusionMatrix, print);
}

map<int, Metrics> C_utils::report(map<pair<int, int>, int> confusionMatrix, bool print){
  map<int, double> precision = precision_score(confusionMatrix);
  map<int, double> accuracy = accuracy_score(confusionMatrix);
  map<int, double> recall = recall_score(confusionMatrix);
  map<int, double> f1score = f1_score(confusionMatrix);
  map<int, double> support = support_score(confusionMatrix);
  map<int, Metrics> report;
  double avg_precision = 0, avg_accuracy = 0, avg_recall = 0, avg_f1score = 0, total_support = 0;

  if(print){
    cout << setw(15) << left  << "class" << setw(15) << left << "precision" << setw(15) << left 
    << "accuracy" << setw(15) << left << "recall" << setw(15) << left << "f1score" << setw(15) << left << "support" << endl;
  }

  for (std::map<int, double>::iterator it = precision.begin(); it != precision.end(); ++it)
  {
    Metrics m;
    m.precision = precision[it->first]; m.accuracy = accuracy[it->first];
    m.recall = recall[it->first]; m.f1score = f1score[it->first];
    m.support = support[it->first];
    report[it->first] = m;
    if(print){
      cout << setw(15) << left << it->first << setw(15) << left << m.precision << setw(15) << left 
      << m.accuracy << setw(15) << left << m.recall << setw(15) << left << m.f1score << setw(15) << left << m.support << endl;
      avg_precision += m.precision*m.support;
      avg_accuracy += m.accuracy*m.support;
      avg_recall += m.recall*m.support;
      avg_f1score += m.f1score*m.support;
      total_support += m.support;
    }
  }
  avg_precision /= total_support;
  avg_accuracy /= total_support;
  avg_recall /= total_support;
  avg_f1score /= total_support;
  
  if(print){
    cout << setw(15) << left << "avg/total" << setw(15) << left << avg_precision << setw(15) << left
    << avg_accuracy << setw(15) << left << avg_recall << setw(15) << left << avg_f1score << setw(15) 
    << left << total_support << endl;  
  }
  
  return report;
}