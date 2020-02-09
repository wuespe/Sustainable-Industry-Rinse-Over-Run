source("setup.R")

train_values = fread("data/train_values.csv")
train_labels = fread("data/train_labels.csv")
### remove target from data
train_values = train_values[phase != "final_rinse"]

### number of processes with a given number of phases
train_values[, .(process_id, phase = factor(phase, levels = c("pre_rinse", "caustic", "intermediate_rinse", "acid")))][
  , .(first_phase = head(phase, 1), 
      last_phase = tail(phase, 1), 
      n_phases = length(unique(phase)))
  , by=process_id][
    , .N, keyby=.(first_phase, last_phase, n_phases)]


## 2. approach - models for multiple phases ------

## pre_rinse : 1
dt_wrk = train_values[phase == "pre_rinse"] %>% aggregate_by_process

dt_train_model1 = dt_wrk %>% prepare_xgb_data

params = list(eta = 0.03, subsample = 0.7, objective = "reg:linear")
xgb_train_model1_cv = xgb.cv(params = params,
                          data = dt_train_model1,
                          nrounds = 300, 
                          nfold = 10, 
                          print_every_n = 10, 
                          early_stopping_rounds = 25)

xgb_train_model1 = xgb.train(params = params, dt_train_model1, nrounds = 60)

# prediction on training set

pred_train_model1 = dt_train_model1 %>% predict(xgb_train_model1, .)
pred_train_model1 = dt_wrk[, .(process_id, pred_model1 = pred_train_model1)]

# prediction on test set
dt_test_model1 = test_values[phase == "pre_rinse"] %>% aggregate_by_process
pred_model1 = dt_test_model1 %>% 
  prepare_xgb_data(add_label = FALSE) %>%
  predict(xgb_train_model1, .)

pred_model1 = dt_test_model1[, .(process_id, pred_model1 = pred_model1)]

## pre_rinse + caustic : 12

## pre_rinse + caustic + intermediate_rinse : 123

## pre_rinse + caustic + intermediate_rinse + acid : 1234

## caustic : 2

## acid : 4