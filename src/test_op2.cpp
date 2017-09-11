#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "utils/c_utils.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

int main(){
	MatrixXd a(3,4);

	a << 1,2,3, 1,
		4,5,6, 1,
		7,8,9, 1;

	VectorXd dataNorm = a.rowwise().squaredNorm().array().sqrt();
	a = a.array().colwise() / dataNorm.array();
	cout << a << endl;

	/*MatrixXd b(a.rows() + a.rows(), a.cols());
	b << a, 
		a;
	cout << b << endl;

	VectorXd c(3);
	c << 1,2,3;

	VectorXd d(c.rows()+ c.rows());
		d << c,c;
	cout << d << endl;*/
	}