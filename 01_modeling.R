source("setup.R")

## Data -----------------

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

# just phase 1 # =3
setdiff(
  train_values[phase == "pre_rinse", unique(process_id)],
  train_values[phase != "pre_rinse", unique(process_id)]
) %>% length

# just phase 2 # =38
setdiff(
  train_values[phase == "caustic", unique(process_id)],
  train_values[phase != "caustic", unique(process_id)]
) %>% length

# just phase 3 # =0

# just phase 4 # =199
setdiff(
  train_values[phase == "acid", unique(process_id)],
  train_values[phase != "acid", unique(process_id)]
) %>% length

## list of processes containing 1. phase
process_ids_1st = train_values[phase == "pre_rinse", unique(process_id)]

## list of processes containing 1 & 2 phases
process_ids_2nd = intersect(
  train_values[phase == "pre_rinse", unique(process_id)],
  train_values[phase == "caustic", unique(process_id)]
)

## list of processes containing 1, 2 & 3 phase
process_ids_3rd = train_values[phase == "pre_rinse", unique(process_id)] %>%
  intersect(train_values[phase == "caustic", unique(process_id)]) %>%
  intersect(train_values[phase == "intermediate_rinse", unique(process_id)])

## list of processes containing 1, 2, 3 & 4 phase
process_ids_4th = train_values[phase == "pre_rinse", unique(process_id)] %>%
  intersect(train_values[phase == "caustic", unique(process_id)]) %>%
  intersect(train_values[phase == "intermediate_rinse", unique(process_id)]) %>%
  intersect(train_values[phase == "acid", unique(process_id)])

# dt_1f[, .N, keyby=pipeline][, dput(pipeline)]
# c("L1", "L2", "L3", "L4", "L6", "L7", "L8", "L9", "L10", "L11", "L12")

# dt_1f = train_values[process_id %in% process_ids_1st]
# dt_1f = dt_1f[phase == "pre_rinse"]

rm(train_values)
### zagregować do poziomu process_id






## models for single phase (pre_rinse, caustic, acid) ----

dt_pre_rinse = train_values[phase == "pre_rinse"]
dt_caustic = train_values[phase == "caustic"]
dt_acid = train_values[phase == "acid"]

## pre_rinse

dt_wrk = dt_pre_rinse %>%
  aggregate_by_process %>%
  prepare_xgb_data

params = list(eta = 0.03, subsample = 0.7, objective = "reg:linear")

xgb_pre_rinse_1_cv = xgb.cv(params = params,
                            data = dt_wrk,
                            nrounds = 200, 
                            nfold = 10, 
                            print_every_n = 10, 
                            early_stopping_rounds = 25)

xgb_pre_rinse_1 = xgb.train(params = params,
                            dt_wrk,
                            nrounds = 90
                            )

## caustic

dt_wrk = dt_caustic %>%
  aggregate_by_process %>%
  prepare_xgb_data

params = list(eta = 0.03, subsample = 0.7, objective = "reg:linear")

xgb_caustic_1_cv = xgb.cv(params = params,
                            data = dt_wrk,
                            nrounds = 200, 
                            nfold = 10, 
                            print_every_n = 10, 
                            early_stopping_rounds = 25)

xgb_caustic_1 = xgb.train(params = params,
                            dt_wrk,
                            nrounds = 150)


## acid

dt_wrk = dt_acid %>%
  aggregate_by_process %>%
  prepare_xgb_data

params = list(eta = 0.03, subsample = 0.7, objective = "reg:linear")

xgb_acid_1_cv = xgb.cv(params = params,
                          data = dt_wrk,
                          nrounds = 200, 
                          nfold = 10, 
                          print_every_n = 10, 
                          early_stopping_rounds = 25)

xgb_acid_1 = xgb.train(params = params,
                          dt_wrk,
                          nrounds = 100)



## Data - test ------------------

test_values = fread("data/test_values.csv")

test_values[order(row_id)][, .(last_phase = tail(phase, 1)), by=.(process_id)][, .N/2967, last_phase]

test_values[, .(process_id, phase = factor(phase, levels = c("pre_rinse", "caustic", "intermediate_rinse", "acid")))][
  , .(first_phase = head(phase, 1), 
      last_phase = tail(phase, 1), 
      n_phases = length(unique(phase)))
  , by=process_id][
    , .N, keyby=.(first_phase, last_phase, n_phases)]


## predictions on test ----

# pre_rinse
pre_rinse_test_data = test_values[phase == "pre_rinse"] %>%
  aggregate_by_process

result = pre_rinse_test_data %>% prepare_xgb_data(add_label = FALSE) %>%
  predict(xgb_pre_rinse_1, .)

pre_rinse_res = data.table(process_id = pre_rinse_test_data$process_id, pre_rinse_res = result)

# caustic
caustic_test_data = test_values[phase == "caustic"] %>%
  aggregate_by_process

result = caustic_test_data %>% prepare_xgb_data(add_label = FALSE) %>%
  predict(xgb_caustic_1, .)

caustic_res = data.table(process_id = caustic_test_data$process_id, caustic_res = result)


# acid
acid_test_data = test_values[phase == "acid"] %>%
  aggregate_by_process

result = acid_test_data %>% prepare_xgb_data(add_label = FALSE) %>%
  predict(xgb_acid_1, .)

acid_res = data.table(process_id = acid_test_data$process_id, acid_res = result)

## combine and average into submission

submission = fread("data/submission_format.csv")

submission = merge(submission, pre_rinse_res, by = "process_id", all.x = TRUE)
submission = merge(submission, caustic_res, by = "process_id", all.x = TRUE)
submission = merge(submission, acid_res, by = "process_id", all.x = TRUE)

submission[1:25, .(pre_rinse_res, 
               caustic_res, 
               acid_res,
               result = (coalesce(pre_rinse_res, 0) + coalesce(caustic_res, 0) + coalesce(acid_res, 0)) / (ifelse(!is.na(pre_rinse_res), 1, 0) + ifelse(!is.na(caustic_res), 1, 0) + ifelse(!is.na(acid_res), 1, 0) )
               )]
submission[, result := (coalesce(pre_rinse_res, 0) + coalesce(caustic_res, 0) + coalesce(acid_res, 0)) / (ifelse(!is.na(pre_rinse_res), 1, 0) + ifelse(!is.na(caustic_res), 1, 0) + ifelse(!is.na(acid_res), 1, 0) )]

submission_1 = submission[, .(process_id, final_rinse_total_turbidity_liter = result)][order(process_id)]

fwrite(submission_1, file = "submission_20190203.csv")

## combine: get result from first completed phase

submission_2 = submission[, .(process_id, final_rinse_total_turbidity_liter = coalesce(pre_rinse_res, caustic_res, acid_res))][order(process_id)]

fwrite(submission_2, file = "submission_20190204.csv")


## combine: get result from last completed phase

submission_3 = submission[, .(process_id, final_rinse_total_turbidity_liter = coalesce(acid_res, caustic_res, pre_rinse_res))][order(process_id)]

fwrite(submission_3, file = "submission_20190205.csv")

