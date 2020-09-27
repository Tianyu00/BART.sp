#' Predicting new observations with a previously fitted wbart_sp model
#'
#' @description
#' This function is used to predict outcomes of new observations with a previously fitted model with type \code{wbart_sp}.
#'
#' @param object An \code{object} of type \code{wbart_sp} (fitted from \code{wbart_sp} function).
#' @param newdata A data frame of covariates to predict for.
#' @param newloc A data frame of location information of the obervations in \code{newdata}.
#' It should have 2 columns (names should be exactly the same as those of the \code{coordinates_train} when fitting the model).
#' @param draw_from_total_distribution Whether sampling from the total distribution when doing prediction on \code{newdata}. If so, it would be slower but utilize all the location information.
#' @param block If \code{draw_from_total_distribution=FALSE}, how many locations should be drawn at the same time.
#' @param seed Setting the seed for reproducibility.
#'
#'
#' @return The return is a list containing these components:
#' @return \code{fhat.test} a matrix of drawings of \eqn{f} corresponding to newdata. Each row corresponds to a draw of the spatial random effect and each column corresponds to a row of \code{newdata}
#' @return \code{yhat.test} a matrix of final predictions (sum of \code{fhat.test} and \code{what.test}) corresponding to newdata. Each row corresponds to a draw of the spatial random effect and each column corresponds to a row of \code{newdata}
#' @return \code{what.test} a matrix of drawings of spatial random effect corresponding to newdata. Each row corresponds to a draw of the spatial random effect and each column corresponds to a row of \code{newdata}.
#' @return \code{unique_test_locations} A data frame of unique test locations in
#' \code{newdata} (order not necessary the same as in \code{newdata}).
predict.wbart_sp <- function(object,
                             newdata,
                             newloc,
                             draw_from_total_distribution,
                             block = 1,
                             seed = 88, ...){
                             # mc.cores = 1, ...) {
  mc.cores=1
  assertthat::assert_that(class(object) == "wbart_sp")
  assertthat::assert_that(is.data.frame(newdata))
  assertthat::assert_that(nrow(newdata) > 0)
  assertthat::assert_that(is.data.frame(newloc))
  assertthat::assert_that(nrow(newloc) == nrow(newdata))
  assertthat::assert_that(ncol(newloc) == 2)
  assertthat::assert_that(all(names(newloc) == names(object$locations_train)))
  assertthat::assert_that(is.logical(draw_from_total_distribution))
  assertthat::assert_that(block > 0 & block %% 1 == 0)
  assertthat::assert_that(is.numeric(seed))
  assertthat::assert_that(mc.cores > 0 & mc.cores %% 1 == 0)

  if (!all(names(object$varcount) %in% names(newdata))) {
    stop("newdata do not have all the variables in the training dataset")
  }
  newdata <- dplyr::select(newdata, all_of(colnames(object$varcount)))

  call <- pwbart

  if (length(object$mu) == 0) object$mu <- object$offset

  nskip <- object$nskip
  ndpost <- object$ndpost
  set.seed(seed)

  if (draw_from_total_distribution | nrow(unique(newloc)) <= block) {
    return(call(newdata, newloc, object$treedraws,
      object$tau2,
      object$logrange,
      object$logsmoothness,
      object$unique_w,
      object$unique_train_locations, object$coordinates_system,
      mc.cores = mc.cores, mu = object$mu, ...
    ))
  } else {
    test_index_list <- create_test_index_list(newloc, block)
    res <- list(
      yhat.test = data.frame(matrix(nrow=ndpost,ncol=0)),
      fhat.test = data.frame(matrix(nrow=ndpost,ncol=0)),
      what.test = data.frame(matrix(nrow=ndpost,ncol=0)),
      unique_w_test = data.frame(matrix(nrow=ndpost,ncol=0)),
      unique_test_locations = data.frame()
    )
    cat(paste(nrow(unique(newloc)), "unique locations,",
              "block size:", block, ',',
              length(test_index_list),'folds\n'))
    for (i in 1:length(test_index_list)) {
      cat(paste('fold', i, '/', length(test_index_list),'\n'))
      newdata_i <- newdata[test_index_list[[i]], ]
      newloc_i <- newloc[test_index_list[[i]], ]
      gc()
      res_i <- call(newdata_i, newloc_i, object$treedraws,
      object$tau2,
      object$logrange,
      object$logsmoothness,
      object$unique_w,
        object$unique_train_locations, object$coordinates_system,
        mc.cores = mc.cores, mu = object$mu, ...
      )
      res$yhat.test <- cbind(res$yhat.test, res_i$yhat.test)
      res$fhat.test <- cbind(res$fhat.test, res_i$fhat.test)
      res$what.test <- cbind(res$what.test, res_i$what.test)
      res$unique_w_test <- cbind(res$unique_w_test, res_i$unique_w_test)
      res$unique_test_locations <- rbind(res$unique_test_locations,
                                         res_i$unique_test_locations)
    }
    res$yhat.test <- res$yhat.test[, Matrix::invPerm(unlist(test_index_list))]
    res$what.test <- res$what.test[, Matrix::invPerm(unlist(test_index_list))]
    return(res)
  }
}
