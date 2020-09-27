#include "my_helper.h"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>
//#include <iostream>

//using namespace Rcpp;

const double log2pi = std::log(2.0 * M_PI);

arma::mat MaternFun(arma::mat distmat, arma::vec covparms) {
  // covparms: 3 parameters: tau^2, logrange, logsmoothness

  double covparms0 = covparms(0);
  double covparms1 = exp(covparms(1));
  double covparms2 = exp(covparms(2));
  if (covparms2>10) {covparms(2)=10;covparms2=10;}

  int d1 = distmat.n_rows;
  int d2 = distmat.n_cols;
  int j1;
  int j2;
  arma::mat covmat(d1,d2);
  double scaledist;

  double normcon = 1 /
    (pow(2.0, covparms2 - 1)* boost::math::tgamma(covparms2));

  for (j1 = 0; j1 < d1; ++j1){
    for (j2 = 0; j2 < d2; ++j2){
      if ( distmat(j1, j2) == 0 ){
        covmat(j1, j2) = 1;
      } else {
        scaledist = distmat(j1, j2)/covparms1;
        covmat(j1, j2) = normcon * pow( scaledist, covparms2 ) *
          boost::math::cyl_bessel_k(covparms2, scaledist);
      }
    }
  }
  if(covmat.has_nan()){covmat.print();covparms.print();}
  if(covmat.has_inf()){covmat.print();covparms.print();}
  covmat = covmat * covparms0;

  return covmat;
}





arma::mat raiz(const arma::mat A){
  arma::vec D;
  arma::mat B;
  arma::eig_sym(D,B,A);

  unsigned int n = D.n_elem;

  arma::mat G(n,n,arma::fill::zeros);

  for(unsigned int i=0;i<n;i++){
      if(D(i)<0 ) D(i)=0;
      G(i,i)=sqrt(D(i));
  }

  return B*G*B.t();
}



arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


arma::vec Mahalanobis(arma::mat x, arma::rowvec center, arma::mat cov) {
    int n = x.n_rows;
    arma::mat x_cen;
    x_cen.copy_size(x);
    for (int i=0; i < n; i++) {
        x_cen.row(i) = x.row(i) - center;
    }
    return sum((x_cen * cov.i()) % x_cen, 1);
}


arma::vec dmvnrm_arma(arma::mat x, arma::rowvec mean, arma::mat sigma, bool log ) {
    arma::vec distval = Mahalanobis(x,  mean, sigma);
    arma::mat tt = arma::symmatu(sigma);
    if(!tt.is_symmetric()) {printf("dmvnrm\n");tt.print();}
    double logdet = sum(arma::log(arma::eig_sym(tt)));
    arma::vec logretval = -( (x.n_cols * log2pi + logdet + distval)/2  ) ;

    if (log) {
        return(logretval);
    } else {
        return(exp(logretval));
    }
}

