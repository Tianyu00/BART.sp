create_Z_matrix_and_distance_matrix <- function(locations_df, coordinates_system) {
  unique_locations <- as.data.frame(unique(locations_df))
  unique_locations$id <- 1:nrow(unique_locations)
  locations_df <- merge(locations_df, unique_locations, sort = FALSE)
  Z <- fastDummies::dummy_columns(locations_df$id)
  z <- as.matrix(Z[, 2:ncol(Z)])

  colnames(z) <- as.integer(substr(
    colnames(z),
    7,
    7 + nchar(nrow(unique_locations))
  ))
  z <- z[, order(as.integer(colnames(z)))]

  if (coordinates_system == "ground") {
    distance <- as.matrix(dist(unique_locations))
  }
  if (coordinates_system == "lonlat") {
    distance <- fields::rdist.earth(x1 = unique_locations, miles = FALSE)
  }
  nn <- nrow(unique_locations)
  locations_df <- as.matrix(locations_df)
  return(list(
    ZMatrix = z,
    distance = distance
  ))
}
