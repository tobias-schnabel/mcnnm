library(fect)

# Set working dir
setwd("/Users/ts/Git/mcnnm/")

# Read the CSV file into an R data frame
long_data <- read.csv("fect_data.csv")

# Start timing
start_time <- Sys.time()

# Run the estimation
results <- fect(Y ~ D, data = long_data, method = "mc", nlambda = 6,
                index = c("id", "time"), force = "two-way")

# End timing
end_time <- Sys.time()

# Calculate elapsed time
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Create a data frame with the results
output <- data.frame(
  att_avg = results$att.avg,
  lambda_cv = results$lambda.cv
)

# Add Y.ct as columns
Y_ct <- as.data.frame(results$Y.ct)
colnames(Y_ct) <- paste0("Y_ct_", seq_len(ncol(Y_ct)))
output <- cbind(output, Y_ct)

# Add elapsed time
output$elapsed_time <- elapsed_time

# Write the results to a CSV file
write.csv(output, "fect_results.csv", row.names = FALSE)
