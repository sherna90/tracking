#ifndef UTILS
#define UTILS

#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
using namespace cv;
using namespace std;

class Performance
{
    private:
        int true_positives, false_positives, false_negatives;
        double avg_precision,avg_recall,ratio;
    public:
        Performance(void);
        double calc(Rect ground_truth, Rect estimate);
        double get_avg_precision(void);
        double get_avg_recall(void);
};



double lnchoose(int  n, int m);
double bhattarchaya(Eigen::VectorXd m1, Eigen::VectorXd m2);
void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void flat(Eigen::MatrixXd& mat);
Eigen::VectorXd average(const Eigen::Ref<const Eigen::MatrixXd>& a,const Eigen::Ref<const Eigen::MatrixXd>& weigths, int axis);
double median(const Eigen::Ref<const Eigen::MatrixXd>& med);
float fastlog2 (float x);
float fastlog (float x);
float fastdigamma (float x);
Eigen::MatrixXd psi(const Eigen::Ref<const Eigen::MatrixXd>& mat);
float psi(float x);
double* linspace(double min, double max, int n);
int positives(const Eigen::Ref<const Eigen::MatrixXd>&  counts);
//int positives(Eigen::VectorXd counts);
double quad_root(double a, double b, double c);
void removeNoTrials(Eigen::MatrixXd& counts);
double trigamma(double x);
Eigen::VectorXd di_pochhammer(double x, const Eigen::Ref<const Eigen::VectorXd>& vec);
Eigen::VectorXd tri_pochhammer(double x, const Eigen::Ref<const Eigen::VectorXd>& vec);
void read_data(const string& filename,Eigen::MatrixXd& data,int rows, int cols);

#endif