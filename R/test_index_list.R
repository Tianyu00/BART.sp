create_test_index_list <- function(loc, block) {
  ret_list <- list()
  unique_loc <- unique(loc)
  unique_loc$id <- 1:nrow(unique_loc)
  loc <- suppressMessages(dplyr::left_join(loc,
                           unique_loc,
                           sorted = FALSE
)
  )
  num_unique_locations_test <- nrow(unique_loc)
  num_groups <- num_unique_locations_test %/% block +
    as.numeric(num_unique_locations_test %% block != 0)
  group_i <- 1
  while (group_i <= num_groups) {
    ret_list[[group_i]] <- which(loc$id %in%
      ((group_i - 1) * block + 1):(group_i * block))
    group_i <- group_i + 1
  }
  return(ret_list)
}
