library(xgboost)

set.seed(14)

train = read.csv('train.csv')
test = read.csv('test.csv')
sample_sub = read.csv('sample_submission.csv')

train_labels = c(seq(3, 9), 12, 14, 15, seq(18, 44), 999)
train_target = match(train$TripType, train_labels) - 1
train_target = data.frame(train$VisitNumber, train_target)
colnames(train_target) = c("A", "B")
train_target = aggregate(train_target$B, list(train_target$A), mean)[, 2]

cat("Convert weekday to one-hot encoding!\n")
train_weekday = model.matrix(~factor(train$Weekday) - 1)
test_weekday = model.matrix(~factor(test$Weekday) - 1)

train_weekday = aggregate(train_weekday, list(train$VisitNumber), mean)[,2 : 8]
test_weekday = aggregate(test_weekday, list(test$VisitNumber), mean)[, 2 : 8]

cat("Get ScanCount!\n")
train_scancount = train$ScanCount
train_scancount[train_scancount > 0] = 0
test_scancount = test$ScanCount
test_scancount[test_scancount > 0] = 0

train_scancount = aggregate(train_scancount, list(train$VisitNumber), mean)[, 2]
test_scancount = aggregate(test_scancount, list(test$VisitNumber), mean)[, 2]

train_return = seq(1, length(train_scancount))
train_buy = seq(1, length(train_scancount))
test_return = seq(1, length(test_scancount))
test_buy = seq(1, length(test_scancount))

for (idx in seq(1, length(train_scancount)))
{
  # cat(idx, "\n")
  if (train_scancount[idx] == 0)
  {
    train_return[idx] = 0
    train_buy[idx] = 1
  }
  else if (train_scancount[idx] == -1)
  {
    train_return[idx] = 1
    train_buy[idx] = 0
  }
  else
  {
    train_return[idx] = 1
    train_buy[idx] = 1
  }
}

for (idx in seq(1, length(test_scancount)))
{
  # cat(idx, "\n")
  if (test_scancount[idx] == 0)
  {
    test_return[idx] = 0
    test_buy[idx] = 1
  }
  else if (test_scancount[idx] == -1)
  {
    test_return[idx] = 1
    test_buy[idx] = 0
  }
  else
  {
    test_return[idx] = 1
    test_buy[idx] = 1
  }
}

train_return_buy = data.frame(train_return, train_buy)
test_return_buy = data.frame(test_return, test_buy)

cat("Convert DepartmentDescription to one-hot encoding!\n")
train_levels = levels(train$DepartmentDescription)
test_levels = levels(test$DepartmentDescription)
levels = unique(c(train_levels, test_levels))

train_depart = cbind(train$VisitNumber, model.matrix(~factor(train$DepartmentDescription, levels) - 1))
test_depart = cbind(test$VisitNumber, model.matrix(~factor(test$DepartmentDescription, levels) - 1))
train_depart = data.frame(train_depart)
test_depart = data.frame(test_depart)

colnames(train_depart)[1] = "VisitNumber"
colnames(test_depart)[1] = "VisitNumber"

train_depart = aggregate(train_depart, list(train_depart$VisitNumber), sum)
test_depart = aggregate(test_depart, list(test_depart$VisitNumber), sum)

train_depart = train_depart[, seq(3, 71)]
test_depart = test_depart[, seq(3, 71)]

train_feat = cbind(train_weekday, train_return_buy, train_depart)
test_feat = cbind(test_weekday, test_return_buy, test_depart)

train_feat = data.frame(train_feat)
test_feat = data.frame(test_feat)

train_data = data.frame(cbind(data.frame(train_target), train_feat))
test_data = data.frame(test_feat)

write.csv(train_data, "train_data.csv", row.names = FALSE)
write.csv(test_data, "test_data.csv", row.names = FALSE)

cat("Train a XGBoost classifier!\n")

cv_idx = sample(nrow(train_feat), 20000)

d_val = xgb.DMatrix(data = data.matrix(train_feat[cv_idx, ]), 
                    label = train_target[cv_idx])
d_train = xgb.DMatrix(data = data.matrix(train_feat[-cv_idx, ]), 
                      label = train_target[-cv_idx])
watchlist = list(val = d_val, train = d_train)

param = list(objective = "multi:softprob",
             eval_metric = "mlogloss",
             eta = 0.01,
             max_depth = 10, 
             subsample = 0.7,
             colsample_bytree = 0.7,
             num_class = 38,
             min_child_weight = 10
)

xgb <- xgb.train(params = param,
                 data = d_train,
                 nrounds = 100000,
                 verbose = 1,
                 early.stop.round = 50,
                 watchlist = watchlist
)



cat("Make predictions!\n")
test_visit = data.frame(test$VisitNumber, test$VisitNumber)
colnames(test_visit) = c("A", "B")
test_visit = aggregate(test_visit$B, list(test_visit$A), mean)[, 2]

test_visit = strtoi(test_visit)

submission <- data.frame(VisitNumber = test_visit)
columns = colnames(sample_sub)
columns = columns[-1]
submission[columns] = NA

test_pred = predict(xgb, data.matrix(test_feat))

submission[columns] = matrix(unlist(test_pred), ncol = 38, byrow = TRUE)

submission$VisitNumber = sample_sub$VisitNumber

cat("Saving the submission file!\n")
write.csv(submission, "xgb.csv", row.names = FALSE)
