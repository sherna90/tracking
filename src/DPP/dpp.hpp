#ifndef DPP_H
#define DPP_H

#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


using namespace cv;
using namespace std;
using namespace Eigen;

class DPP
{
public:
	DPP();
	vector<Rect> run(vector<Rect> preDetections, const VectorXd &detectionWeights, const VectorXd &penaltyWeights,MatrixXd &featureValues,  double lambda,  double mu, double epsilon);
	//vector<Rect> run(vector<Rect> preDetections, VectorXd &detectionWeights,VectorXd &penaltyWeights, MatrixXd &featureValues,
	//VectorXd &qualityTermResults, double lambda, double mu, double epsilon);
	VectorXd likelihood(VectorXd &qualityTerm, MatrixXd &similarityTerm);

private:
	VectorXd get_quality_term(const VectorXd &detectionWeights,const  VectorXd &nPenalty, double lambda);
	MatrixXd get_similarity_term(MatrixXd &featureValues, MatrixXd &intersectionArea, MatrixXd &sqrtArea, double mu);
	vector<int> solve(VectorXd &qualityTerm, MatrixXd &similarityTerm, double epsilon);
	MatrixXd squared_exponential_kernel(MatrixXd X, double nu, double sigma_f);
};

/*template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
	const ArgType &m_arg;
	const RowIndexType &m_rowIndices;
	const ColIndexType &m_colIndices;
public:
	typedef Matrix<typename ArgType::Scalar,
		RowIndexType::SizeAtCompileTime,
		ColIndexType::SizeAtCompileTime,
		ArgType::Flags&RowMajorBit?RowMajor:ColMajor,
		RowIndexType::MaxSizeAtCompileTime,
		ColIndexType::MaxSizeAtCompileTime> MatrixType;
	indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
	: m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
	{}
	const typename ArgType::Scalar& operator() (Index row, Index col) const {
		return m_arg(m_rowIndices[row], m_colIndices[col]);
	}
};

template <class ArgType, class RowIndexType, class ColIndexType>
CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>, typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
	typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;

	LLT<Matrix<Scalar,Dynamic,Dynamic>> lltOfM(M);
	if(lltOfM.info() == Eigen::NumericalIssue)
  {
		PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
	else
	{
		auto& U = lltOfM.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
	}
  return ld;
}
*/
#endif
