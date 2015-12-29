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


cat("Convert DepartmentDescription to one-hot encoding for buy and return!\n")
## buy
train_levels = levels(train$DepartmentDescription)
test_levels = levels(test$DepartmentDescription)
levels = unique(c(train_levels, test_levels))

train_depart = cbind(train$VisitNumber, model.matrix(~factor(train$DepartmentDescription, levels) - 1))
test_depart = cbind(test$VisitNumber, model.matrix(~factor(test$DepartmentDescription, levels) - 1))
train_depart = data.frame(train_depart)
test_depart = data.frame(test_depart)

colnames(train_depart)[1] = c("VisitNumber")
colnames(test_depart)[1] = c("VisitNumber")

train_depart = aggregate(train_depart, list(train_depart$VisitNumber), sum)
test_depart = aggregate(test_depart, list(test_depart$VisitNumber), sum)

train_depart = train_depart[, seq(3, 71)]
test_depart = test_depart[, seq(3, 71)]

cat("Fineline Number!\n")
train$FinelineNumber[is.na(train$FinelineNumber)] = -1
test$FinelineNumber[is.na(test$FinelineNumber)] = -1
train_fln_levels = levels(factor(train$FinelineNumber))
test_fln_levels = levels(factor(test$FinelineNumber))
fln_levels = unique(c(train_fln_levels, test_fln_levels))

train_fln = cbind(train$VisitNumber, model.matrix(~factor(train$FinelineNumber, fln_levels) - 1))
test_fln = cbind(test$VisitNumber, model.matrix(~factor(test$FinelineNumber, fln_levels) - 1))

train_fln = train$ScanCount * train_fln
test_fln = test$ScanCount * train_fln

# train_fln = data.frame(train_fln)
# test_fln = data.frame(test_fln)

colnames(train_fln)[1] = c("VisitNumber")
colnames(test_fln)[1] = c("VisitNumber")

train_fln = aggregate(train_fln, list(train_fln$VisitNumber), sum)
test_fln = aggregate(test_fln, list(test_fln$VisitNumber), sum)

train_fln = train_fln[, seq(3, 71)]
test_fln = test_fln[, seq(3, 71)]


cat("Group features!\n")

train_feat = cbind(train_weekday, train_depart, train_fln)
test_feat = cbind(test_weekday, test_depart, test_fln)

train_feat = data.frame(train_feat)
test_feat = data.frame(test_feat)

# train_data = data.frame(cbind(data.frame(train_target), train_feat))
# test_data = data.frame(test_feat)

# write.csv(train_data, "train_data.csv", row.names = FALSE)
# write.csv(test_data, "test_data.csv", row.names = FALSE)

cat("Train a XGBoost classifier!\n")

cv_idx = sample(nrow(train_feat), 10000)

d_val = xgb.DMatrix(data = data.matrix(train_feat[cv_idx, ]), 
                    label = train_target[cv_idx])
d_train = xgb.DMatrix(data = data.matrix(train_feat[-cv_idx, ]), 
                      label = train_target[-cv_idx])
watchlist = list(val = d_val, train = d_train)

param = list(objective = "multi:softprob",
             eval_metric = "mlogloss",
             eta = 0.02,
             max_depth = 10, 
             subsample = 0.7,
             colsample_bytree = 0.7,
             num_class = 38,
             min_child_weight = 1
             # num_parallel_tree = 5
)

xgb <- xgb.train(params = param,
                 data = d_train,
                 nrounds = 100000,
                 early.stop.round = 100,
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

