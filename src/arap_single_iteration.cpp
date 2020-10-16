#include "../include/arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U)
{
  const int d = bc.cols();
  const int r = K.rows()/d;
  // Compute covariance matrices
  Eigen::MatrixXd S = K*U;
  Eigen::MatrixXd R(r*d,d);
  for(int k = 0;k<r;k++)
  {
    Eigen::Matrix3d Sk = S.block(k*3,0,3,3);
    // Rescaling is important. Perhaps this should be performed in polar_svd3x3
    Sk = Sk.array()/Sk.array().abs().maxCoeff();
    Eigen::Matrix3d Rk;
    igl::polar_svd3x3(Sk,Rk);
    R.block(k*3,0,3,3) = Rk;
  }
  Eigen::MatrixXd B = -K.transpose()*R;
  igl::min_quad_with_fixed_solve(
    data,B,bc,Eigen::MatrixXd(),U);
}
