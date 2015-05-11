#ifndef UTILS
#define UTILS

#include <stdlib.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdint.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
using namespace cv;

class Performance
{
    private:
        Rect intersection;
        int true_positives, false_positives, false_negatives;
        double avg_precision,avg_recall,ratio;
    public:
        Performance(void);
        void calc(Rect ground_truth, Rect estimate);
        double get_avg_precision(void);
        double get_avg_recall(void);
};



double lnchoose(int  n, int m);
double bhattarchaya(Eigen::VectorXd m1, Eigen::VectorXd m2);
void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void flat(Eigen::MatrixXd& mat);
Eigen::VectorXd average(Eigen::MatrixXd a,Eigen::MatrixXd weigths, int axis);
double median(Eigen::MatrixXd med);
float fastlog2 (float x);
float fastlog (float x);
float fastdigamma (float x);
Eigen::MatrixXd psi(Eigen::MatrixXd mat);
float psi(float x);
double* linspace(double min, double max, int n);
int positives(Eigen::MatrixXd& counts);
int positives(Eigen::VectorXd counts);
double quad_root(double a, double b, double c);
void removeNoTrials(Eigen::MatrixXd& counts);
double trigamma(double x);



#endif