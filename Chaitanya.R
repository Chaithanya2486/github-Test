# Load required libraries
library(data.table)
library(stringr)

# Load your cleaned dataset (assuming you've already done make.names and gsub)
train <- fread("train.csv")
test  <- fread("test.csv")

# Clean the names as before
clean_names <- make.names(colnames(train), unique = TRUE)
clean_names <- gsub("\\.+", "_", clean_names)
colnames(train) <- clean_names
colnames(test)  <- clean_names

# Step 1: Strip index suffixes to find base names
base_names <- str_replace(clean_names, "_[0-9]+$", "")

# Step 2: Find duplicated base names (conflicting under formulas)
duplicated_bases <- base_names[duplicated(base_names)]

# Step 3: Get all column names corresponding to those base names
conflict_cols <- clean_names[base_names %in% duplicated_bases]
print(conflict_cols)

# Step 4: Drop conflicting columns
train <- train[, !colnames(train) %in% conflict_cols, with = FALSE]
test  <- test[, !colnames(test) %in% conflict_cols, with = FALSE]

# Step 5: Confirm result
cat("Remaining columns:", length(colnames(train)), "\n")
anyDuplicated(make.names(colnames(train), unique = TRUE))  

# Ensure factor type
train$Activity <- as.factor(train$Activity)
test$Activity  <- as.factor(test$Activity)

# Safe feature list
features <- setdiff(colnames(train), c("subject", "Activity"))
length(features)

library(ranger)
library(parallel)

# Subset helper
get_subset_data <- function(data, selected) {
  selected_features <- features[selected]
  data[, c(selected_features, "Activity"), with = FALSE]
}

# Optimised fitness function
fitness_function <- function(selected_vec) {
  if (sum(selected_vec) == 0) return(Inf)
  
  selected_features <- features[selected_vec]
  
  X_train <- train[, ..selected_features]
  y_train <- train$Activity
  
  X_test  <- test[, ..selected_features]
  y_test  <- test$Activity
  
  model <- ranger(dependent.variable.name = "Activity",
                  data = data.frame(Activity = y_train, X_train),
                  write.forest = TRUE,
                  classification = TRUE,
                  probability = TRUE,
                  num.threads = parallel::detectCores())
  
  preds <- predict(model, data = data.frame(X_test))$predictions
  pred_class <- colnames(preds)[apply(preds, 1, which.max)]
  
  # Match predicted class labels to factor levels of y_test
  pred_num <- match(pred_class, levels(y_test))
  actual_num <- as.numeric(y_test)
  
  rmse <- sqrt(mean((actual_num - pred_num)^2))
  return(rmse)
}

set.seed(123)
rand_select <- sample(0:1, length(features), replace = TRUE, prob = c(0.8, 0.2))
fitness_function(rand_select) 

# --------------------------
# Load required libraries
# --------------------------
library(GA)
library(pso)
library(DEoptim)
library(caret)
library(ggplot2)
library(doParallel)
library(ranger)
library(data.table)

# --------------------------
# Register parallel backend
# --------------------------
registerDoParallel(cores = parallel::detectCores())

# --------------------------
# Top 100 High-Variance Features
# --------------------------
variances <- sapply(train[, !c("subject", "Activity"), with = FALSE], var)
top_features <- names(sort(variances, decreasing = TRUE))[1:100]
features <- top_features
cat("Top 100 high-variance features selected.\n")

# --------------------------
# GA optimisation
# --------------------------
run_ga_with_early_stop <- function(fitness_fn, features, max_generations = 30, stop_patience = 5) {
  best_rmse_history <- c()
  generation <- 1
  same_count <- 0
  last_rmse <- NA
  best_solution <- NULL
  
  repeat {
    cat("Running Generation:", generation, "\n")
    
    ga_result <- ga(
      type = "binary",
      fitness = function(x) -fitness_fn(x),
      nBits = length(features),
      popSize = 20,
      maxiter = 1,
      run = 1,
      parallel = TRUE,
      suggestions = if (!is.null(best_solution)) best_solution
    )
    
    best_solution <- ga_result@population
    current_rmse <- -ga_result@fitnessValue
    best_rmse_history <- c(best_rmse_history, current_rmse)
    
    if (!is.na(last_rmse) && abs(current_rmse - last_rmse) < 1e-6) {
      same_count <- same_count + 1
    } else {
      same_count <- 1
      last_rmse <- current_rmse
    }
    
    cat("Best RMSE:", current_rmse, "| Same Count:", same_count, "\n\n")
    
    if (same_count >= stop_patience || generation >= max_generations) {
      break
    }
    
    generation <- generation + 1
  }
  
  return(list(ga_result = ga_result, history = best_rmse_history))
}

# Run GA
set.seed(100)
ga_run <- run_ga_with_early_stop(fitness_function, features, max_generations = 30, stop_patience = 5)
ga_result <- ga_run$ga_result
ga_rmse_history <- ga_run$history
selected_ga <- which(ga_result@solution[1, ] == 1)
selected_features_ga <- features[selected_ga]
cat("GA selected", length(selected_features_ga), "features\n")

# --------------------------
# PSO optimisation
# --------------------------
pso_wrapper <- function(par) {
  binary <- ifelse(par > 0.5, 1, 0)
  fitness_function(binary)
}

set.seed(101)
pso_result <- psoptim(
  par = rep(0.5, length(features)),
  fn = pso_wrapper,
  lower = rep(0, length(features)),
  upper = rep(1, length(features)),
  control = list(maxit = 30, s = 20, trace = 1)
)

selected_features_pso <- features[which(pso_result$par > 0.5)]
cat("PSO selected", length(selected_features_pso), "features\n")

# --------------------------
# Define fitness function
# --------------------------
fitness_function <- function(x) {
  if (sum(x) == 0) return(Inf)  # Avoid empty feature sets
  selected <- features[which(x == 1)]
  
  model <- ranger(
    dependent.variable.name = "Activity",
    data = data.frame(Activity = train$Activity, train[, ..selected]),
    write.forest = TRUE,  # ✅ Required for predict()
    classification = TRUE,
    probability = FALSE,
    num.threads = 1
  )
  
  preds <- predict(model, data = data.frame(test[, ..selected]))$predictions
  rmse <- sqrt(mean((as.numeric(preds) - as.numeric(test$Activity))^2))
  return(rmse)
}

# --------------------------
# DE optimisation
# --------------------------
de_wrapper <- function(x) {
  binary <- ifelse(x > 0.5, 1, 0)
  fitness_function(binary)
}

set.seed(102)
de_result <- DEoptim(
  fn = de_wrapper,
  lower = rep(0, length(features)),
  upper = rep(1, length(features)),
  control = DEoptim.control(NP = 30, itermax = 30, parallelType = 0, trace = TRUE)
)

selected_features_de <- features[which(de_result$optim$bestmem > 0.5)]
cat("DE selected", length(selected_features_de), "features\n")

# --------------------------
# Model evaluation function
# --------------------------
evaluate_model <- function(feature_set, name) {
  model <- ranger(
    dependent.variable.name = "Activity",
    data = data.frame(Activity = train$Activity, train[, ..feature_set]),
    write.forest = TRUE,
    classification = TRUE,
    probability = TRUE,
    num.threads = parallel::detectCores()
  )
  
  preds <- predict(model, data = data.frame(test[, ..feature_set]))$predictions
  pred_class <- colnames(preds)[apply(preds, 1, which.max)]
  cm <- confusionMatrix(factor(pred_class, levels = levels(test$Activity)), test$Activity)
  cat(paste0("\n--- ", name, " ---\n"))
  print(cm$overall)
  return(cm$overall['Accuracy'])
}

# --------------------------
# Evaluate accuracies
# --------------------------
acc_ga <- evaluate_model(selected_features_ga, "GA")
acc_pso <- evaluate_model(selected_features_pso, "PSO")
acc_de <- evaluate_model(selected_features_de, "DE")

# --------------------------
# Accuracy comparison plot
# --------------------------
accuracy_df <- data.frame(
  Method = c("GA", "PSO", "DE"),
  Accuracy = c(acc_ga, acc_pso, acc_de)
)

ggplot(accuracy_df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = round(Accuracy, 4)), vjust = -0.3) +
  ylim(0, 1) +
  theme_minimal() +
  labs(title = "Accuracy Comparison: GA vs PSO vs DE")
