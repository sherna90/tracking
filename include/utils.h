#ifndef UTILS
#define UTILS

#include <stdlib.h>
#include <cmath>
#include <Eigen/Dense>

double lnchoose(int  n, int m);
double bhattarchaya(Eigen::VectorXd m1, Eigen::VectorXd m2);
void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void flat(Eigen::MatrixXd& mat);
Eigen::VectorXd average(Eigen::MatrixXd a,Eigen::MatrixXd weigths, int axis);
double median(Eigen::MatrixXd med);
static inline float fastlog2 (float x);
static inline float fastlog (float x);
static inline float fastdigamma (float x);
Eigen::MatrixXd psi(Eigen::MatrixXd mat);
float psi(float x);

#endif