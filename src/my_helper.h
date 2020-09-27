

#ifndef GUARD_my_helper
#define GUARD_my_helper


#include <RcppArmadillo.h>
//#include <cmath>

// https://stackoverflow.com/questions/47274696/segment-fault-when-using-rcpp-armadillo-and-openmp-prarallel-with-user-defined-f
arma::mat MaternFun(arma::mat distmat, arma::vec covparms);

// https://stackoverflow.com/questions/41884478/why-cant-i-get-the-square-root-of-this-symmetric-positive-definite-matrix-in-ar
arma::mat raiz(const arma::mat A);

// https://gallery.rcpp.org/articles/simulate-multivariate-normal/
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);

arma::vec Mahalanobis(arma::mat x, arma::rowvec center, arma::mat cov);

arma::vec dmvnrm_arma(arma::mat x, arma::rowvec mean, arma::mat sigma, bool log = TRUE);
#endif


