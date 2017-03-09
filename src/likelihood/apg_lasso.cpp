//Author: Diego Vergara
#include "apg_lasso.hpp"

APG_LASSO::APG_LASSO(){
	init = false;
}

APG_LASSO::APG_LASSO(int _iterations, double _beta, double _lambda){
	iterations = _iterations;
	beta = _beta;
	lambda = _lambda;
	init = true;
}

//Objective function: f(x) + lambda*norm1(x)
double APG_LASSO::objetive_Function(MatrixXd &A, VectorXd &x, VectorXd &b){
	double Of = 0.0;
	if (init){
		Of = function(A,x,b) + lambda * (x.array().abs()).colwise().sum()(0);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return Of;
}

//f(x) = (1/2)||Ax-b||^2
double APG_LASSO::function(MatrixXd &A, VectorXd &x, VectorXd &b){
	double Ax_b = 0.0;
	if (init){
		VectorXd aux_vec = matrixDot(A,x).array() -b.array();
		Ax_b = 0.5*aux_vec.transpose().dot(aux_vec);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return Ax_b;
}

//gradient of f(x)
VectorXd APG_LASSO::gradient(MatrixXd &A, VectorXd &x, VectorXd &b){
	VectorXd grad(A.cols());
	if (init){	
		VectorXd aux_vector = matrixDot(A,x) - b;
		MatrixXd aux_matrix = A.transpose();
		grad = matrixDot(aux_matrix,aux_vector);
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return grad;
}

//Model function evaluated at x and touches f(x) in xk_acc
double APG_LASSO::modelFunction(VectorXd &x, VectorXd &xk, MatrixXd &A, VectorXd &b, double step_lenght){
	double model = 0.0;
	if (init){
		VectorXd xDiff = x -xk;
		double innerProd = gradient(A,xk,b).transpose().dot(xDiff);
		model = function(A, xk, b) + innerProd + (1.0 / (2.0*step_lenght)) * (xDiff.transpose().dot(xDiff));
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return model;
}

//Shrinkage or Proximal operation, Soft Thresholding
VectorXd APG_LASSO::softThresholding(VectorXd &x, double gamma){
	VectorXd prox_vect(x.rows());
	if (init){	
		VectorXd vec_abs = x.array().abs()- gamma;
		//prox_vect = sign(x).cwiseProduct(vecMax(0.0, vec_abs));
		prox_vect = sign(x).array() *(vecMax(0.0, vec_abs)).array();
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return prox_vect;
}

VectorXd APG_LASSO::matrixDot(MatrixXd &A, VectorXd &x){
	VectorXd aux(A.rows());
	for (int i = 0; i < A.rows(); ++i)
	{
		aux[i] = A.row(i).dot(x);
	}
	return aux;
}

VectorXd APG_LASSO::sign(VectorXd &x){
	VectorXd s(x.rows());
	for (int i = 0; i < x.rows(); ++i){
		if (x[i] > 0.0){
			s[i] = 1.0;
		}
		else if(x[i] < 0.0){
			s[i] = -1.0;
		}
		else{
			s[i] = 0.0;
		}
	}
	return s;
}

VectorXd APG_LASSO::vecMax(double value, VectorXd &vec){
	VectorXd vector_max(vec.rows());
	for (int i = 0; i < vec.rows(); ++i){
		vector_max[i] = max(value, vec[i]);
	}
	return vector_max;
}

void APG_LASSO::fit(MatrixXd &A, VectorXd &b, double _step_lenght){
	if (init){	
		//For Accelerated Proximal Gradient Descent
 		cout << setprecision(10) << fixed;
 		VectorXd xk_acc = VectorXd::Zero(A.cols());
 		VectorXd yk_acc = xk_acc;
 		double tk_acc = 1.0;
 		double Dobj_acc = 0.0;
 		VectorXd x_kplus1_acc = VectorXd::Zero(A.cols());
 		//=================== Accelerated Proximal GD ========================
 		for (int i = 0; i < iterations; ++i){
 			
 			double step_lenght = _step_lenght;
 			//Line search
			while (true){
				//Accelerated Gradient Descent (GD) Step
				x_kplus1_acc = yk_acc - step_lenght*gradient(A,yk_acc,b);
				if (function(A,x_kplus1_acc,b) <= modelFunction(x_kplus1_acc,xk_acc, A, b, step_lenght)){
					break;
				}
				else{
					step_lenght = beta * step_lenght;
				}
			}

 			//Proximal Operation (Shrinkage)
 			x_kplus1_acc = softThresholding(x_kplus1_acc, step_lenght*lambda);
 			double t_kplus1_acc = 0.5 + 0.5*sqrt(1+ 4*(pow(tk_acc,2)));
 			VectorXd y_kplus1_acc = x_kplus1_acc + ((tk_acc -1) / (tk_acc +1)) * (x_kplus1_acc - xk_acc);

 			//Change in the value of objective function for this iteration
 			Dobj_acc = fabs(objetive_Function(A, xk_acc, b)) -(objetive_Function(A, x_kplus1_acc, b));

 			//Update
 			xk_acc = x_kplus1_acc;
 			yk_acc = y_kplus1_acc;
 			tk_acc = t_kplus1_acc;

 			//cout << Dobj_acc << endl;
 			//Terminating Condition        
 			if (Dobj_acc < 1.0)
 			{
 				break;
 			}
 			
 		}
 		weights = xk_acc;

	}
	else{
		cout << "Error: No initialized run APG_LASSO"<< endl;
	}
}

VectorXd APG_LASSO::predict(){
	VectorXd predict;
	if (init)
	{			
		predict = weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return predict;
}

