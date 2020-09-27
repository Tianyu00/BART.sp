
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017-2018 Robert McCulloch and Rodney Sparapani

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

pwbart <- function(
                   x.test, # x matrix to predict at
                   x.loc,
                   treedraws, # $treedraws from wbart
                   tau2s,
                   logrange,
                   logsmoothnesses,
                   w,
                   unique_train_locs,
                   coordinates_system,
                   mu = 0, # mean to add on
                   mc.cores = 1L, # thread count
                   transposed = FALSE,
                   nice = 19L # mc.pwbart only
) {
  if (!transposed) x.test <- t(bartModelMatrix(x.test))

  if (nrow(unique(unique_train_locs)) ==
      nrow(unique(rbind(x.loc, unique_train_locs)))) {
    new_loc <- FALSE
    locations_all <- unique_train_locs
    if (coordinates_system == "ground") {
      distance <- as.matrix(dist(unique_train_locs))
    }
    if (coordinates_system == "lonlat") {
      distance <- fields::rdist.earth(
        x1 = unique_train_locs,
        miles = FALSE
      )
    }
    distance_all <- distance
    ttest.locations <- suppressMessages(dplyr::left_join(x.loc,
                                        unique_train_locs,
                                        sort = FALSE
      )
    )
    ttest.locations <- rbind(ttest.locations, unique_train_locs)
    ttest.Z <- fastDummies::dummy_columns(ttest.locations$id)
    ttest.Z <- as.matrix(ttest.Z[, 2:ncol(ttest.Z)])
    colnames(ttest.Z) <- as.integer(substr(
      colnames(ttest.Z),
      7,
      7 + nchar(nrow(unique_train_locs))
    ))
    ttest.Z <- ttest.Z[, order(as.integer(colnames(ttest.Z)))]

    ttest.Z <- ttest.Z[1:nrow(x.loc), ]
    z2 <- ttest.Z
    unique.locations.all <- unique_train_locs
  } else {
    new_loc <- TRUE
    unique.locations.all <- rbind(
                                 suppressMessages(dplyr::anti_join(unique(x.loc),
                                                   unique_train_locs,
                                                   sort = FALSE
                                                   )),
      unique_train_locs
    )
    locations_all <- unique.locations.all
    if (coordinates_system == "ground") {
      distance_all <- as.matrix(dist(unique.locations.all))
    }
    if (coordinates_system == "lonlat") {
      distance_all <- fields::rdist.earth(
        x1 = unique.locations.all,
        miles = FALSE
      )
    }

    unique.locations.all$id <- 1:nrow(unique.locations.all)
    ttest.locations <- suppressMessages(dplyr::left_join(x.loc, unique.locations.all,
                                        sort = FALSE))
    if (coordinates_system == "ground") {
      ttest.locations.2 <- rbind(
        ttest.locations,
        data.frame(
          x = rep(0, nrow(unique.locations.all)),
          y = rep(0, nrow(unique.locations.all)),
          id = 1:nrow(unique.locations.all)
        )
      )
    }
    if (coordinates_system == "lonlat") {
      ttest.locations.2 <- rbind(
        ttest.locations,
        data.frame(
          lon = rep(0, nrow(unique.locations.all)),
          lat = rep(0, nrow(unique.locations.all)),
          id = 1:nrow(unique.locations.all)
        )
      )
    }

    ttest.Z.2 <- fastDummies::dummy_columns(ttest.locations.2$id)
    ttest.Z.2 <- as.matrix(ttest.Z.2[, 2:ncol(ttest.Z.2)])
    colnames(ttest.Z.2) <- as.integer(substr(
      colnames(ttest.Z.2),
      7,
      7 + nchar(nrow(unique.locations.all))
    ))
    ttest.Z.2 <- ttest.Z.2[, order(as.integer(colnames(ttest.Z.2)))]

    z2 <- ttest.Z.2[1:nrow(ttest.locations), ]
  }

  res <- .Call(
    "cpwbart",
    treedraws, # trees list
    x.test, # the test x
    tau2s,
    logrange,
    logsmoothnesses,
    new_loc,
    w,
    distance_all,
    nrow(unique.locations.all), # nn, #total uni locs
    # nn2, #uni. test loc
    nrow(unique.locations.all) - nrow(unique_train_locs),
    mc.cores # thread count
  )

  if (new_loc) {
    w_all <- cbind(res$w_test, w)
    what <- w_all %*% t(z2)
  } else {
    what <- w %*% t(z2)
  }
  locations_in_test_set_index <- which(colSums(z2)>0)

    return(list(
      yhat.test = res$yhat.test + mu + what,
      fhat.test = res$yhat.test + mu ,
      what.test = what,
      unique_w_test = w_all[,locations_in_test_set_index],
      unique_test_locations = unique.locations.all[locations_in_test_set_index,
                                                 1:2]
    ))
}
