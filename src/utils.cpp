/**
 * @file utils.cpp
 * @brief utilities
 * @author Guillermo Becerra
 */
#include "../include/utils.hpp"


using namespace Eigen;

double lnchoose(int  n, int m){
    double nf=lgamma(n+1);
    double mf=lgamma(m+1);
    double nmmnf = lgamma(n-m+1);
    return (nf-(mf+nmmnf));

}

double bhattarchaya(VectorXd m1, VectorXd m2){
    RowVectorXd coef;
    coef=m1.cwiseProduct(m2).cwiseSqrt()/sqrt(m1.sum()*m2.sum());
    return sqrt(1-coef.sum());
}

void removeRow(MatrixXd& matrix, unsigned int rowToRemove){
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
    
}

void flat(MatrixXd& mat){
    mat.resize(mat.rows()*mat.cols(),1);
    
}

VectorXd average(MatrixXd a,MatrixXd weigths, int axis){
    MatrixXd r = a;
    flat(weigths);
    
    if(axis==0){

        for(int i=0;i<r.cols();i++){
            r.col(i)= r.col(i).array()*weigths.array(); 
        }
        return (1/weigths.sum())*r.colwise().sum(); 
    
    }else if(axis==1){

        for(int i=0;i<r.rows();i++){
            r.row(i)= r.row(i).array()*weigths.transpose().array(); 
        }
        return (1/weigths.sum())*r.rowwise().sum().transpose();
    
    }else{
        //cout << "invalid argument on average function AVERAGE" << endl;
        exit(EXIT_FAILURE);
    }
    
}
double median(MatrixXd med){
    MatrixXd aux = med;
    flat(aux);
    int n = aux.rows();
    if(n%2==0){
        return (aux(n/2,0)+aux(n/2-1,0))/2.0;

    }else{
        return aux(n/2,0);
    }
}
// Utils for digamma from http://fastapprox.googlecode.com/svn/trunk/fastapprox/src/fastonebigheader.h

float fastlog2 (float x)
{
  union { float f; uint32_t i; } vx = { x };
  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f 
           - 1.72587999f / (0.3520887068f + mx.f);
}

float fastlog (float x)
{
  return 0.69314718f * fastlog2 (x);
}


float fastdigamma (float x)
{
  float twopx = 2.0f + x;
  float logterm = fastlog (twopx);

  return (-48.0f + x * (-157.0f + x * (-127.0f - 30.0f * x))) /
         (12.0f * x * (1.0f + x) * twopx * twopx)
         + logterm;
}

MatrixXd psi(MatrixXd mat){
    MatrixXd res(mat.rows(),mat.cols());

    for(int i=0;i<mat.rows();i++){
        for(int j=0;j<mat.cols();j++){
            res(i,j)=fastdigamma(mat(i,j));
        }
    }
    return res;
}

float psi(float x){
    
    return fastdigamma(x);
}

double* linspace(double min, double max, int n){
    double* result=new double[n];
    int iterator = 0;
    for (int i = 0; i <= n-2; i++){
        double temp = min + i*(max-min)/(floor((double)n) - 1);
        result[iterator]=temp;
        iterator += 1;
    }
    result[iterator]=max;
    return result;
}


Performance::Performance(void){
    avg_precision=0.0;avg_recall=0.0;
}
void Performance::calc(Rect ground_truth, Rect estimate){
    intersection=ground_truth & estimate;
    true_positives=0;false_positives=0;false_negatives=0;
    ratio = double(intersection.area())/double(ground_truth.area());
    if(ratio==1.0){ 
        true_positives=ground_truth.area();
        false_negatives=0;
        false_positives=0;
    }
    else if(ratio>1.0){
        true_positives=ground_truth.area();
        false_negatives=0;
        false_positives=estimate.area()-ground_truth.area();   
    }
    else if(ratio<1.0){
        true_positives=intersection.area();
        false_negatives=ground_truth.area()-intersection.area();
        estimate.area()>0?false_positives=estimate.area()-intersection.area():false_positives=1;   
    }
    avg_precision+=double(true_positives)/double(true_positives+false_positives); 
    avg_recall+=double(true_positives)/double(true_positives+false_negatives);
}
double Performance::get_avg_precision(void){
    return avg_precision;
}
double Performance::get_avg_recall(void){
    return avg_recall;
}
