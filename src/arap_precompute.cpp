#include "../include/arap_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/arap_linear_block.h>
#include <igl/cotmatrix.h>

void arap_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data,
  Eigen::SparseMatrix<double> & K)
{
  // Compute cotangents opposite each half-edge
  Eigen::MatrixXd C;
  igl::cotmatrix_entries(V,F,C);

  const int n = V.rows();
  const int m = F.rows();
  const int r = V.rows();
  const int d = 3;

  std::vector<Eigen::Triplet<double> > LIJV;
  std::vector<Eigen::Triplet<double> > KIJV;
  // Loop over faces
  for(int f = 0;f<m;f++)
  {
    // Loop over edges
    for(int c = 0;c<3;c++)
    {
      // Vertex indices
      const int i = F(f,(c+1)%3);
      const int j = F(f,(c+2)%3);
      const double Cfc = C(f,c);
      Eigen::RowVectorXd diff  = Cfc*(V.row(i)-V.row(j));
      // Throw edge-quantity at each rotation group of which the edge is a
      // member
      std::vector<int> edge_sets = { F(f,0), F(f,1), F(f,2) };
      for(const auto k : edge_sets)
      {
        // Throw into L
        LIJV.emplace_back(i,i,-Cfc);
        LIJV.emplace_back(j,j,-Cfc);
        LIJV.emplace_back(i,j, Cfc);
        LIJV.emplace_back(j,i, Cfc);
        // Throw into K
        KIJV.emplace_back(d*k+0,i, diff(0));
        KIJV.emplace_back(d*k+0,j,-diff(0));
        KIJV.emplace_back(d*k+1,i, diff(1));
        KIJV.emplace_back(d*k+1,j,-diff(1));
        KIJV.emplace_back(d*k+2,i, diff(2));
        KIJV.emplace_back(d*k+2,j,-diff(2));
      }
    }
  }
  Eigen::SparseMatrix<double> L(n,n);
  L.setFromTriplets(LIJV.begin(),LIJV.end());
  K.resize(r * d,n);
  K.setFromTriplets(KIJV.begin(),KIJV.end());

  igl::min_quad_with_fixed_precompute(
    (-L).eval(),b,Eigen::SparseMatrix<double>(),true,data);
}
