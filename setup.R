library(data.table)
library(magrittr)
library(hutils)
library(xgboost)

## funkcje ----------------
one_hot_encode = function(dt, cols) {
  setDT(dt)
  newDT = copy(dt)
  newDT[, oheID := .I]
  melted = melt(newDT, measure.vars = cols, value.factor = TRUE)
  
  encoded_cols = dcast(melted, oheID~variable+value, fun.aggregate = length, drop = FALSE)
  
  newDT = merge(newDT[, !cols, with=FALSE], encoded_cols, by="oheID")
  newDT[, !"oheID"]
}

slope_coef = function(y) {
  x = 1:length(y)
  sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))**2)
}

aggregate_by_process = function(dt) {
  newDT = dt[, .(process_id = unique(process_id))]
  cols_to_agg = setdiff(colnames(dt), c("process_id", "row_id", "object_id"))
  for(colname in cols_to_agg) {
    if(is.numeric(dt[[colname]])) {
      transformations_names = paste(colname, c("min", "max", "std", "avg", "avg_l5", "trend"), sep = "_")
      transformations_expr = list(min, max, sd, mean, . %>% tail(5) %>% mean, slope_coef)
      exprs = setNames(transformations_expr, transformations_names)
      column = parse(text = colname)
      
      new_cols = dt[, lapply(exprs, . %>% do.call(list(eval(column)))), by = process_id]
      
      newDT = merge(newDT, new_cols, by = "process_id")
    }
    
  }
  dt_pipeline = dt[, .N, by = .(process_id, pipeline)][, N := NULL][, pipeline := factor(pipeline, levels = c("L1", "L2", "L3", "L4", "L6", "L7", "L8", "L9", "L10", "L11", "L12"))]
  dt_pipeline = dt_pipeline %>% one_hot_encode(cols = "pipeline")
  merge(newDT, dt_pipeline, by = "process_id")
}


prepare_xgb_data = function(dt, add_label = TRUE) {
  if(add_label) {
    newDT = merge(dt, train_labels, by = "process_id")
    res = xgb.DMatrix(as.matrix(newDT[, !c("process_id", "final_rinse_total_turbidity_liter")]), label = newDT$final_rinse_total_turbidity_liter)
  } else {
    res = xgb.DMatrix(as.matrix(dt[, !c("process_id")]))
  }
  
  res
}