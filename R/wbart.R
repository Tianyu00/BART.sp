
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

wbart <- function(x_train, y_train, coordinates_train,
                  x_test = matrix(0.0, 0, 0), coordinates_test = NULL,
                  sparse = FALSE, theta = 0, omega = 1,
                  a = 0.5, b = 1, augment = FALSE, rho = NULL,
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
                  logrange_select_sd  , logsmoothness_select_sd  ,
                  sigma2_prior_a ,
                  sigma2_prior_b ,
                  tau2_prior_a ,
                  tau2_prior_b ,
                  logrange_init ,
                  logsmoothness_init ,
                  tau2_init ,
                  logrange_prior_mean ,
                  logrange_prior_sd ,
                  logsmoothness_prior_mean ,
                  logsmoothness_prior_sd ,
                  coordinates_system) {

  ##################### locations ###################
  Z_matrix_and_distance_matrix <- create_Z_matrix_and_distance_matrix(
    coordinates_train,
    coordinates_system
  )
  ZMatrix <- Z_matrix_and_distance_matrix[[1]]
  distanceMatrix <- Z_matrix_and_distance_matrix[[2]]
  rm(Z_matrix_and_distance_matrix)
  gc()

  #--------------------------------------------------
  # data
  n <- length(y_train)

  if (!transposed) {
    temp <- bartModelMatrix(x_train, numcut,
      usequants = usequants,
      cont = cont, xinfo = xinfo, rm.const = rm.const
    )
    x_train <- t(temp$X)
    numcut <- temp$numcut
    xinfo <- temp$xinfo
    if (length(x_test) > 0) {
      x_test <- bartModelMatrix(x_test)
      x_test <- t(x_test[, temp$rm.const])
    }
    rm.const <- temp$rm.const
    grp <- temp$grp
    rm(temp)
  } else {
    rm.const <- NULL
    grp <- NULL
  }

  if (n != ncol(x_train)) {
    stop(paste(
      "The length of y_train and the number of rows in x_train must",
      "be identical"
    ))
  }

  p <- nrow(x_train)
  np <- ncol(x_test)
  if (length(rho) == 0) rho <- p
  if (length(rm.const) == 0) rm.const <- 1:p
  if (length(grp) == 0) grp <- 1:p

  ## if(p>1 & length(numcut)==1) numcut=rep(numcut, p)

  y_train <- y_train - fmean
  #--------------------------------------------------
  # set nkeeps for thinning
  if ((nkeeptrain != 0) & ((ndpost %% nkeeptrain) != 0)) {
    nkeeptrain <- ndpost
    cat("*****nkeeptrain set to ndpost\n")
  }
  if ((nkeeptest != 0) & ((ndpost %% nkeeptest) != 0)) {
    nkeeptest <- ndpost
    cat("*****nkeeptest set to ndpost\n")
  }
  if ((nkeeptestmean != 0) & ((ndpost %% nkeeptestmean) != 0)) {
    nkeeptestmean <- ndpost
    cat("*****nkeeptestmean set to ndpost\n")
  }
  if ((nkeeptreedraws != 0) & ((ndpost %% nkeeptreedraws) != 0)) {
    nkeeptreedraws <- ndpost
    cat("*****nkeeptreedraws set to ndpost\n")
  }
  #--------------------------------------------------
  # prior
  nu <- sigdf
  if (is.na(lambda)) {
    if (is.na(sigest)) {
      if (p < n) {
        df <- data.frame(t(x_train), y_train)
        lmf <- lm(y_train ~ ., df)
        sigest <- summary(lmf)$sigma
      } else {
        sigest <- sd(y_train)
      }
    }
    qchi <- qchisq(1.0 - sigquant, nu)
    lambda <- (sigest * sigest * qchi) / nu # lambda parameter for sigma prior
  } else {
    sigest <- sqrt(lambda)
  }

  if (is.na(sigmaf)) {
    tau <- (max(y_train) - min(y_train)) / (2 * k * sqrt(ntree))
  } else {
    tau <- sigmaf / sqrt(ntree)
  }
  #--------------------------------------------------
  ptm <- proc.time()

  # call
  res <- .Call("cwbart",
    PACKAGE = "BART.sp",
    n, # number of observations in training data
    p, # dimension of x
    np, # number of observations in test data
    x_train, # pxn training data x
    y_train, # pxn training data x
    x_test, # p*np test data x
    ntree,
    numcut,
    ndpost * keepevery,
    nskip,
    power,
    base,
    tau,
    nu,
    lambda,
    sigest,
    w,
    sparse,
    theta,
    omega,
    grp,
    a,
    b,
    rho,
    augment,
    nkeeptrain,
    nkeeptest,
    nkeeptestmean,
    nkeeptreedraws,
    printevery,
    ZMatrix,
    distanceMatrix,
    xinfo,
    logrange_select_sd,
    logsmoothness_select_sd,
    sigma2_prior_a,
    sigma2_prior_b,
    tau2_prior_a,
    tau2_prior_b,
    logrange_init,
    logsmoothness_init,
    tau2_init,
    logrange_prior_mean,
    logrange_prior_sd,
    logsmoothness_prior_mean[1],
    logsmoothness_prior_sd[1]
  )

  res$nskip <- nskip
  res$ndpost <- ndpost
  res$sigma_all <- res$sigma
  res$sigma <- res$sigma[(nskip+1):(nskip+ndpost)]
  res$unique_w = res$what.train
  res$unique_train_locations <- unique(coordinates_train)
  res$what.train <- res$what.train %*% t(ZMatrix)
  res$what.test <- NULL
  res$proc.time <- proc.time() - ptm
  res$coordinates_system <- coordinates_system
  res$mu <- fmean
  res$fhat.train <- res$yhat.train + fmean
  res$fhat.test <- NULL

  print(dim(res$what.train))
  res$yhat.train <- res$yhat.train + res$what.train
  res$yhat.test <- NULL
  if (nkeeptreedraws > 0) {
    names(res$treedraws$cutpoints) <- dimnames(x_train)[[1]]
  }
  dimnames(res$varcount)[[2]] <- as.list(dimnames(x_train)[[1]])
  dimnames(res$varprob)[[2]] <- as.list(dimnames(x_train)[[1]])
  ## res$nkeeptreedraws=nkeeptreedraws
  #   res$varcount.mean <- apply(res$varcount, 2, mean)
  #   res$varprob.mean <- apply(res$varprob, 2, mean)
  res$rm.const <- rm.const
  attr(res, "class") <- "wbart_sp"

  return(res)
}
