# Load required library
library(dplyr)
library(readr)

# Load training and test datasets
# -----------------------------
train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

# Data preprocessing: Impute missing values and scale features
# -----------------------------

# Train set:
# - Impute missing Age values by the median within each Sex group
# - Standardize numeric columns (Pclass, Age, SibSp, Parch, Fare)
train <- train %>%
  group_by(Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup() %>%
  mutate(across(c(Pclass, Age, SibSp, Parch, Fare), scale))

cat("Train dataset: Missing Age values imputed with group median and features standardized.\n")
print(head(train$Age))

# Test set:
# - Impute missing Age values by the median within each Sex group
# - Impute missing Fare values by the overall median
# - Standardize numeric columns (Pclass, Age, SibSp, Parch, Fare)
test <- test %>%
  group_by(Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup() %>%
  mutate(Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare)) %>%
  mutate(across(c(Pclass, Age, SibSp, Parch, Fare), scale))

cat("Test dataset: Missing Age and Fare values imputed, features standardized.\n")
print(head(test$Age))
print(head(test$Fare))

# Model training: Logistic regression
# -----------------------------
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
             data = train,
             family = binomial)

cat("Logistic regression model fitted successfully.\n")
print(summary(model))

# Evaluate model on training data
# -----------------------------
train$pred_prob <- predict(model, newdata = train, type = "response")
train$pred_class <- ifelse(train$pred_prob > 0.5, 1, 0)

accuracy <- mean(train$pred_class == train$Survived)
cat(paste0("Training Accuracy: ", round(accuracy * 100, 2), "%\n"))

# Generate predictions on test data
# -----------------------------
test$pred_survived <- predict(
  model,
  newdata = test[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")],
  type = "response"
)

# Save results
# -----------------------------
write_csv(test[, c("PassengerId", "pred_survived")], file = "R_app/R_predictions.csv")
cat("Predictions saved to 'R_app/R_predictions.csv'.\n")
cat("Titanic survival prediction pipeline completed successfully.\n")

# run in src folder
# docker build -t titanic-app-r -f R_app/Dockerfile .
# docker run --rm -v ${PWD}:/workspace -w /workspace titanic-app-r Rscript R_app/app.R