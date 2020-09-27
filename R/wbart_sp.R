#' Spatially Adjusted Bayesian Additive Regression Trees for continuoues outcomes
#'
#' @description
#' \code{wbart_sp} is a Bayesian "sum-of-trees" model designed for spatial data. It is
#' built upon the original BART model (see ...) with an extra spatial random effect.
#'
#' For a numeric continuous outcome \eqn{y}, we have \eqn{y = f(x) + e_s + e}, where \eqn{e_s} is the spatial random effect and \eqn{e ~ N(0, sigma^2)}. See ...
#'
#' @details
#' \code{wbart_sp} is the only function (besides S3 method \code{predict.wbart_sp}) provided by
#' this package.
#'
#' \code{wbart_sp} implements the spatially adjusted Bayesian Additive Regression Trees (in single
#'  thread or multiple threads). S3 method \code{predict.wbart_sp} implements the prediction of a model of class wbart_sp.
#'
#' The detailed information about the model please see: paper/github ...
#'
#' @param x_train
#' Explanatory variables for training (in sample) data.
#' Must be a data frame, with rows corresponding to observations and columns to variables.
#' If a variable is a factor in a data frame, it is replaced with dummies. Note that q dummies are created if q>2 and one dummy is created if q=2, where q is the number of levels of the factor.
#' Location information can be either in \code{x_train} or not but must be in \code{coordinates_train}.
#' @param y_train Continuous dependent variable for training (in sample) data. Must be a vector whose length equals to the number of rows in \code{x_train}.
#' @param coordinates_train The location information of observations in \code{x_train}. Must be a dataframe of 2 columns and the same number of rows as \code{x_train}. If latitude and longitude are provided as location information, the 2 columns of \code{coordinates_train} must be named exactly 'lon' and 'lat' (order matter) and the argument \code{coordiantes} set as 'lonlat'.
#'  If locations information on the ground are provided, the 2 columns must be named exactly 'x' and 'y' (order matters) and argument \code{coordiantes} set as 'ground'.
#'  The distane between locations is calculated accordingly (see \code{coordinates_system}).
#' @param coordiantes_system What the \code{coordinates_train} are. Must be either 'lonlat' or 'ground'. If \code{coordinates_train} is the longitude and latitude information, \code{coordinates_train} should be 'lonlat' and the distance is calculated using grand circle distance with unit km. If \code{coordinates_train} is the location information on the ground, \code{coordinates_train} should be 'ground' and the distance is calculated using Euclidean distance.
#' @param x_test Explanatory variables for test (out of sample) data. Should have same structure as \code{x_train} (Must be a data frame, with rows corresponding to observations and columns to variables). If provided, must also provide \code{coordinates_test}.
#' @param coordinates_test The location information of observations in \code{x_test}. Should have same structure as \code{coordinates_train}. It must be of the same kind of coordinate system as \code{coordinates_train} and named exactly the same as \code{coordinates_train} (order matters).
#'
#'
#'
#' @param sparse Argument for code\{wbart} in r package BART, please see documentation there.
#' @param theta Argument for code\{wbart} in r package BART, please see documentation there.
#' @param omega Argument for code\{wbart} in r package BART, please see documentation there.
#' @param a Argument for code\{wbart} in r package BART, please see documentation there.
#' @param b Argument for code\{wbart} in r package BART, please see documentation there.
#' @param rho Argument for code\{wbart} in r package BART, please see documentation there.
#' @param augment Argument for code\{wbart} in r package BART, please see documentation there.
#' @param xinfo Argument for code\{wbart} in r package BART, please see documentation there.
#' @param usequants Argument for code\{wbart} in r package BART, please see documentation there.
#' @param cont Argument for code\{wbart} in r package BART, please see documentation there.
#' @param rm.const Argument for code\{wbart} in r package BART, please see documentation there.
#' @param sigest Argument for code\{wbart} in r package BART, please see documentation there.
#' @param sigdf Argument for code\{wbart} in r package BART, please see documentation there.
#' @param sigquant Argument for code\{wbart} in r package BART, please see documentation there.
#' @param k Argument for code\{wbart} in r package BART, please see documentation there.
#' @param power Argument for code\{wbart} in r package BART, please see documentation there.
#' @param base Argument for code\{wbart} in r package BART, please see documentation there.
#' @param sigmaf Argument for code\{wbart} in r package BART, please see documentation there.
#' @param lambda Argument for code\{wbart} in r package BART, please see documentation there.
#' @param fmean Argument for code\{wbart} in r package BART, please see documentation there.
#' @param w Argument for code\{wbart} in r package BART, please see documentation there.
#' @param ntree Argument for code\{wbart} in r package BART, please see documentation there.
#' @param numcut Argument for code\{wbart} in r package BART, please see documentation there.
#' @param ndpost Argument for code\{wbart} in r package BART, please see documentation there.
#' @param nskip Argument for code\{wbart} in r package BART, please see documentation there.
#' @param keepevery Argument for code\{wbart} in r package BART, please see documentation there.
#' @param nkeeptrain Argument for code\{wbart} in r package BART, please see documentation there.
#' @param nkeeptest Argument for code\{wbart} in r package BART, please see documentation there.
#' @param nkeeptestmean Argument for code\{wbart} in r package BART, please see documentation there.
#' @param nkeeptreedraws Argument for code\{wbart} in r package BART, please see documentation there.
#' @param printevery Argument for code\{wbart} in r package BART, please see documentation there.
#' @param transposed Argument for code\{wbart} in r package BART, please see documentation there.
#' @param seed Argument for code\{wbart} in r package BART, please see documentation there
#'
#'
#'
#' @param logrange_select_sd Spatial residual sampling parameter. logRange select SD in mcmc. (see ...)
#' @param logsmoothness_select_sd Spatial residual sampling parameter. logSmoothness select SD in mcmc. (see ...)
#' @param sigma2_prior_a Prior paramter for the random noise \eqn{e}. (see ...)
#' @param sigma2_prior_b Prior paramter for the random noise \eqn{e}. (see ...)
#' @param tau2_prior_a Prior paramter for the matern correlation function tau2 (see ...)
#' @param tau2_prior_b Prior paramter for the matern correlation function tau2 (see ...)
#' @param logrange_init Initial value for logrange in mcmc.
#' @param logsmoothness_init Initial value for logsmoothness in mcmc.
#' @param tau2_init Initial value for tau2 in mcmc.
#' @param logrange_prior_mean Prior paramter for the matern correlation function logrange mean (see ...)
#' @param logrange_prior_sd Prior paramter for the matern correlation function logrange sd (see ...)
#' @param logsmoothness_prior_mean Prior paramter for the matern correlation function logsmoothness mean (see ...)
#' @param logsmoothness_prior_sd Prior paramter for the matern correlation function logsmoothness sd (see ...)
#' @param mc Whether fitting the model in parallel. (which usually improves the model performance but requires multiple cores.) Please also set the number of threads in argument \code{mc.cores}.
#' @param mc.cores How many threads to use if fitting the model in parallel. If \code{mc=FALSE}, this argument does not matter. If \code{mc=TRUE}, how many threads to use.
#' @param draw_from_total_distribution If draw from total distribution or ? distribution in the prediction. If no \code{x_test}, does not matter. Usually it would be slower but perserving and utilizing all the location information in the testing dataset to set \code{draw_from_total_distribution=TRUE} instead of \code{FALSE}.
#' @param block The spatial random effect of how many locations to predict at one time if \code{draw_from_total_distribution=TRUE}. If \code{draw_from_total_distribution=FALSE}, \code{block} is not used.
#'
#'
#' @return \code{wbart_sp} returns an object of type \code{wbart_sp} which is a list. It has the following components:
#' @return  \code{fhat.train} A matrix with ndpost rows and nrow(x_train) columns. Each row corresponds to a draw \eqn{f^*}{f*} from the posterior of \eqn{f} and each column corresponds to a row of x_train. The \eqn{(i,j)} value is \eqn{f^*(x)}{f*(x)} for the \eqn{i^{th}}{i\^th} kept draw of \eqn{f} and the \eqn{j^{th}}{j\^th} row of x.train. Burn-in is dropped. NOTICE: this is the not final prediction value, \code{yhat.train} is.
#' @return \code{fhat.test} Same as \code{fhat.train} but now the x's are the rows of the test data.
#' @return  \code{yhat.train} A matrix with ndpost rows and nrow(x_train) columns. Each row corresponds to the final prediction (sum of a draw from \eqn{f(x)} and a draw of the spatial random effect) and each column corresponds to a row of x_train.
#' @return \code{yhat.test} Same as \code{yhat.train} but now the x's are the rows of the test data.
#' @return  \code{what.train} A matrix with ndpost rows and nrow(x_train) columns. Each row corresponds to a draw of the spatial random effect and each column corresponds to a row of x_train.
#' @return \code{what.test} Same as \code{what.train} but now the x's are the rows of the test data.
#' @return \code{sigma} post burn in draws of sigma, length = ndpost.
#' @return \code{sigma_all} A data frame of burn in draws and post burn in draws of sigma, dim = (nskip + ndpost/mc.cores) * (\code{mc.cores}). Can be used to inspect convergence.
#' @return \code{tau2} post burn in draws of sigma, length = ndpost.
#' @return \code{logrange} post burn in draws of logrange, length = ndpost.
#' @return \code{logsmoothness} post burn in draws of logsmoothness, length = ndpost.
#' @return \code{nskip} nskip
#' @return \code{ndpost} ndpost
#' @return \code{mu} mean of y_train
#' @return \code{varcount} a matrix with ndpost rows and nrow(x_train) columns. Each row is for a draw. For each variable (corresponding to the columns), the total count of the number of times that variable is used in a tree decision rule (over all trees) is given.
#' @return \code{varprob} a matrix with ndpost rows and nrow(x_train) columns. Each row is for a draw. For each variable (corresponding to the columns), the probability (frequency / total frequency) that variable is used in a tree decision rule (over all trees) is given.
#' @return \code{sigest} The rough error standard deviation (\eqn{\sigma}{sigma}) used in the prior.
#' @return \code{coordinates_system} Coordiantes parameters for \code{coordinates_train}.
#' @return \code{unique_train_lcations} Unique locations in the training data.
#' @return \code{unique_w} sampled spatial random effects according for \code{unique_train_locations}.
#' @return \code{proc.time} processing time
#'
##' @examples
## See vignettes.
#' @references
#' Chipman, H., George, E., and McCulloch R. (2010)
#' Bayesian Additive Regression Trees.
#' \emph{The Annals of Applied Statistics}, \bold{4,1}, 266-298 <doi:10.1214/09-AOAS285>.
#'
#' Chipman, H., George, E., and McCulloch R. (2006)
#' Bayesian Ensemble Learning.
#' Advances in Neural Information Processing Systems 19,
#' Scholkopf, Platt and Hoffman, Eds., MIT Press, Cambridge, MA, 265-272.
#'
#' Friedman, J.H. (1991)
#' Multivariate adaptive regression splines.
#' \emph{The Annals of Statistics}, \bold{19}, 1--67.
#'
#' Linero, A.R. (2018)
#' Bayesian regression trees for high dimensional prediction and variable
#' selection. \emph{JASA}, \bold{113}, 626--36.
wbart_sp <- function(x_train, y_train, coordinates_train,
                     coordinates_system,
                     x_test = matrix(0, 0, 0, 0), coordinates_test = NULL,
                     sparse = FALSE, theta = 0, omega = 1,
                     a = 0.5, b = 1, rho = NULL, augment = FALSE,
                     xinfo = matrix(0.0, 0, 0), usequants = FALSE,
                     cont = FALSE, rm.const = TRUE,
                     sigest = NA, sigdf = 3, sigquant = .90,
                     k = 2.0, power = 2.0, base = .95,
                     sigmaf = NA, lambda = NA,
                     fmean = mean(y_train),
                     w = rep(1, length(y_train)),
                     ntree = 200L, numcut = 100L,
                     ndpost = 1000L, nskip = 100L, keepevery = 1L,
                     nkeeptrain = ndpost, nkeeptest = ndpost,
                     nkeeptestmean = ndpost, nkeeptreedraws = ndpost,
                     printevery = 100L, transposed = FALSE,
                     # new arguments
                     logrange_select_sd = 0.3, logsmoothness_select_sd = 0.3,
                     sigma2_prior_a = 10.0,
                     sigma2_prior_b = 1.0,
                     tau2_prior_a = 1.0,
                     tau2_prior_b = 1.0,
                     logrange_init = 0.0,
                     logsmoothness_init = 0.0,
                     tau2_init = 1.0,
                     logrange_prior_mean = 1.0,
                     logrange_prior_sd = 0.5,
                     logsmoothness_prior_mean = 0.0,
                     logsmoothness_prior_sd = 0.5,
                     mc, mc.cores=2,
                     draw_from_total_distribution = TRUE, block = 50,
                     seed = 88) {
  assertthat::assert_that(is.data.frame(x_train))
  assertthat::assert_that(nrow(x_train) > 0)
  assertthat::assert_that(is.vector(y_train))
  assertthat::assert_that(nrow(x_train) == length(y_train))
  assertthat::assert_that(is.data.frame(coordinates_train))
  assertthat::assert_that(nrow(x_train) == nrow(coordinates_train))
  assertthat::assert_that(ncol(coordinates_train) == 2)
  assertthat::assert_that(coordinates_system %in% c("lonlat", "ground"))
  if (coordinates_system == "lonlat") {
    assertthat::assert_that(all(names(coordinates_train) == c("lon", "lat")))
  } else {
    assertthat::assert_that(all(names(coordinates_train) == c("x", "y")))
  }
  assertthat::assert_that(is.logical(sparse))
  assertthat::assert_that(is.numeric(theta))
  assertthat::assert_that(theta >= 0)
  assertthat::assert_that(is.numeric(omega))
  assertthat::assert_that(omega > 0)
  assertthat::assert_that(is.numeric(a))
  assertthat::assert_that(a > 0)
  assertthat::assert_that(is.numeric(b))
  assertthat::assert_that(b > 0)
  assertthat::assert_that(is.logical(augment))
  if (!is.null(rho)) {
    assertthat::assert_that(rho >= 0)
  }
  # xinfo
  assertthat::assert_that(is.logical(usequants))
  assertthat::assert_that(is.logical(cont))
  assertthat::assert_that(is.logical(rm.const))
  if (!is.na(sigest)) assertthat::assert_that(sigest > 0)
  assertthat::assert_that(sigdf > 0 & sigdf %% 1 == 0)
  assertthat::assert_that(sigquant >= 0 & sigquant <= 1)
  assertthat::assert_that(k > 0)
  assertthat::assert_that(power > 0)
  assertthat::assert_that(base > 0)
  if (!is.na(sigmaf)) assertthat::assert_that(sigmaf > 0)
  if (!is.na(lambda)) assertthat::assert_that(lambda > 0)
  assertthat::assert_that(is.numeric(fmean))
  assertthat::assert_that(is.vector(w))
  assertthat::assert_that(length(w) == nrow(x_train))
  assertthat::assert_that(ntree > 0 & ntree %% 1 == 0)
  # numcut
  assertthat::assert_that(ndpost > 0 & ndpost %% 1 == 0)
  assertthat::assert_that(nskip > 0 & nskip %% 1 == 0)
  assertthat::assert_that(keepevery > 0 & keepevery %% 1 == 0)
  assertthat::assert_that(nkeeptrain >= 0 & nkeeptrain <= ndpost)
  assertthat::assert_that(nkeeptest >= 0 & nkeeptest <= ndpost)
  assertthat::assert_that(nkeeptestmean >= 0 & nkeeptestmean <= ndpost)
  assertthat::assert_that(nkeeptreedraws >= 0 & nkeeptreedraws <= ndpost)
  assertthat::assert_that(printevery > 0 & printevery %% 1 == 0)
  assertthat::assert_that(is.logical(transposed))
  assertthat::assert_that(is.numeric(logrange_select_sd) & logrange_select_sd > 0)
  assertthat::assert_that(is.numeric(logsmoothness_select_sd) &
    logsmoothness_select_sd > 0)
  assertthat::assert_that(is.numeric(sigma2_prior_a) & sigma2_prior_a > 0)
  assertthat::assert_that(is.numeric(sigma2_prior_b) & sigma2_prior_b > 0)
  assertthat::assert_that(is.numeric(tau2_prior_a) & tau2_prior_a > 0)
  assertthat::assert_that(is.numeric(tau2_prior_b) & tau2_prior_b > 0)
  assertthat::assert_that(is.numeric(logrange_init))
  assertthat::assert_that(is.numeric(logsmoothness_init))
  assertthat::assert_that(is.numeric(logrange_prior_mean))
  assertthat::assert_that(is.numeric(logrange_prior_sd) & logrange_prior_sd > 0)
  assertthat::assert_that(is.numeric(logsmoothness_prior_mean))
  assertthat::assert_that(is.numeric(logsmoothness_prior_sd)
  & logsmoothness_prior_sd > 0)
  assertthat::assert_that(is.logical(mc))
  assertthat::assert_that(mc.cores > 0 & mc.cores %% 1 == 0)
  assertthat::assert_that(is.logical(draw_from_total_distribution))
  assertthat::assert_that(block > 0 & block %% 1 == 0)
  assertthat::assert_that(is.numeric(seed))

  if (length(x_test) > 0) {
    assertthat::assert_that(is.data.frame(x_test))
    assertthat::assert_that(is.data.frame(coordinates_test))
    assertthat::assert_that(nrow(x_test) == nrow(coordinates_test))
    assertthat::assert_that(ncol(coordinates_test) == 2)
    assertthat::assert_that(all(names(coordinates_test) == names(coordinates_train)))
  }

  if (!mc) {
    res <- wbart(
      x_train = x_train, y_train = y_train,
      coordinates_train = coordinates_train,
      x_test = x_test, coordinates_test = coordinates_test,
      sparse = sparse, theta = theta, omega = omega,
      a = a, b = b, augment = augment, rho = rho,
      xinfo = xinfo, usequants = usequants,
      cont = cont, rm.const = rm.const,
      sigest = sigest, sigdf = sigdf, sigquant = sigquant,
      k = k, power = power, base = base,
      sigmaf = sigmaf, lambda = lambda,
      fmean = fmean,
      w = w,
      ntree = ntree, numcut = numcut,
      ndpost = ndpost, nskip = nskip, keepevery = keepevery,
      nkeeptrain = nkeeptrain, nkeeptest = nkeeptest,
      nkeeptestmean = nkeeptestmean, nkeeptreedraws = nkeeptreedraws,
      printevery = printevery, transposed = transposed,
      # new arguments
      logrange_select_sd = logrange_select_sd,
      logsmoothness_select_sd = logsmoothness_select_sd,
      sigma2_prior_a = sigma2_prior_a,
      sigma2_prior_b = sigma2_prior_b,
      tau2_prior_a = tau2_prior_a,
      tau2_prior_b = tau2_prior_b,
      logrange_init = logrange_init,
      logsmoothness_init = logsmoothness_init,
      tau2_init = tau2_init,
      logrange_prior_mean = logrange_prior_mean,
      logrange_prior_sd = logrange_prior_sd,
      logsmoothness_prior_mean = logsmoothness_prior_mean,
      logsmoothness_prior_sd = logsmoothness_prior_sd,
      coordinates_system = coordinates_system
    )
  } else {
    res <- mc.wbart(
      x_train = x_train, y_train = y_train,
      mc.cores = mc.cores,
      coordinates_train = coordinates_train,
      x_test = x_test, coordinates_test = coordinates_test,
      sparse = sparse, theta = theta, omega = omega,
      a = a, b = b, augment = augment, rho = rho,
      xinfo = xinfo, usequants = usequants,
      cont = cont, rm.const = rm.const,
      sigest = sigest, sigdf = sigdf, sigquant = sigquant,
      k = k, power = power, base = base,
      sigmaf = sigmaf, lambda = lambda,
      fmean = fmean,
      w = w,
      ntree = ntree, numcut = numcut,
      ndpost = ndpost, nskip = nskip, keepevery = keepevery,
      printevery = printevery, transposed = transposed,
      # new arguments
      logrange_select_sd = logrange_select_sd,
      logsmoothness_select_sd = logsmoothness_select_sd,
      sigma2_prior_a = sigma2_prior_a,
      sigma2_prior_b = sigma2_prior_b,
      tau2_prior_a = tau2_prior_a,
      tau2_prior_b = tau2_prior_b,
      logrange_init = logrange_init,
      logsmoothness_init = logsmoothness_init,
      tau2_init = tau2_init,
      logrange_prior_mean = logrange_prior_mean,
      logrange_prior_sd = logrange_prior_sd,
      logsmoothness_prior_mean = logsmoothness_prior_mean,
      logsmoothness_prior_sd = logsmoothness_prior_sd,
      coordinates_system = coordinates_system
    )
  }
  if (length(x_test) > 0) {
    predictions <- predict(
      object = res,
      newdata = x_test,
      newloc = coordinates_test,
      draw_from_total_distribution =
        draw_from_total_distribution,
      block = block,
      seed = seed
    )
    res$yhat.test <- predictions$yhat.test
    res$fhat.test <- predictions$fhat.test
    res$what.test <- predictions$what.test
  }

  return(res)
}
