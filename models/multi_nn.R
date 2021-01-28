library(openxlsx)
library(pROC)
library(ROCR)
library(stringr)
library(jsonlite)
library(ggplot2)

load("/home/chzze/01_Rscript/33_SSC/RData/33_00_ftlist.RData")


### 1. Dataset: training, validation, test, test2 set ####
df_train = openxlsx::read.xlsx('/data/SNUBH/SSC/info/dataset/exp111_train.xlsx', detectDates=TRUE)
df_val = openxlsx::read.xlsx('/data/SNUBH/SSC/info/dataset/exp111_val.xlsx', detectDates=TRUE)

df_test = openxlsx::read.xlsx('/data/SNUBH/SSC/info/dataset/exp131_test.xlsx', detectDates=TRUE)
df_test2 = openxlsx::read.xlsx('/data/SNUBH/SSC/info/dataset/exp151_test23.xlsx', detectDates=TRUE)

### **a. Preprocessing ####
df_train$PATIENT_AGE = 0.01 * df_train$PATIENT_AGE
df_train$VAS_MED = 0.1 * df_train$VAS_MED

df_val$PATIENT_AGE = 0.01 * df_val$PATIENT_AGE
df_val$VAS_MED = 0.1 * df_val$VAS_MED

df_test$PATIENT_AGE = 0.01 * df_test$PATIENT_AGE
df_test$VAS_MED = 0.1 * df_test$VAS_MED

df_test2$PATIENT_AGE = 0.01 * df_test2$PATIENT_AGE
df_test2$VAS_MED = 0.1 * df_test2$VAS_MED


### 2. Fit logistic regression based on clinical variables ####
### **a. Fitting ####
clv = c('PATIENT_AGE', 'VAS_MED', 'TRAUMA', 'DOMINANT_SIDE')
clv_var = paste(clv, collapse='+')
lgs_clv_formula = as.formula(paste('SSC_LABEL_BIN1 ~', clv_var))

lgs_clv_fit = glm(lgs_clv_formula, family='binomial', data=df_train)
summary(lgs_clv_fit)


sigmoid = function(x) {
  sig = 1/(1+exp(-x))
  return(sig)
}


### **b. Predict ####
pred_clv_train = predict(lgs_clv_fit, newdata=df_train, type='response')
pred_clv_val = predict(lgs_clv_fit, newdata=df_val, type='response')
pred_clv_test = predict(lgs_clv_fit, newdata=df_test, type='response')
pred_clv_test2 = predict(lgs_clv_fit, newdata=df_test2, type='response')


### **c. Calculate AUC ####
roc(df_train$SSC_LABEL_BIN1, pred_clv_train)  # AUC: 0.6458
roc(df_val$SSC_LABEL_BIN1, pred_clv_val)  # AUC: 0.5747
roc(df_test$SSC_LABEL_BIN1, pred_clv_test)  # AUC: 0.6401
roc(df_test2$SSC_LABEL_BIN1, pred_clv_test2)  # AUC: 0.6769

pre_train_path = '/data/SNUBH/SSC/exp153/Model58'
model_path = '/data/SNUBH/SSC/exp151/Model58/Model58'

 
merge_mm_df = function(model_path, pre_serial, num_weight, lgs_clv_fit, pre_df, data_npy) {
  # model_path = '/data/SNUBH/SSC/exp153/Model58'
  result_serial = paste0('result-', sprintf('%03d', pre_serial))
  pre_img_df = paste0(paste('Model58', data_npy, sprintf('%03d', pre_serial), sprintf('%03d', num_weight), sep='_'), '.csv')

  dl_pre_df = read.csv(paste(model_path, result_serial, pre_img_df, sep='/'))
  mm_df = data.frame(dl_pre_df, NN_LOGIT=predict(lgs_clv_fit, newdata=pre_df, type='link'))
  return(mm_df)
}


m_ntrain = merge_mm_df(pre_train_path, 0, 1, lgs_clv_fit, df_train, 'ntrain.npy')
m_val = merge_mm_df(model_path, 0, 3, lgs_clv_fit, df_val, 'trval.npy')
m_test1 = merge_mm_df(model_path, 0, 3, lgs_clv_fit, df_test, 'test.npy')
m_test2 = merge_mm_df(model_path, 0, 3, lgs_clv_fit, df_test2, 'test23.npy')


### **d. Save multi-modal coefficient ####
pre_train_mm = function(pre_serial, lgs_clv_fit, df_train) {
  
  pre_train_path = '/data/SNUBH/SSC/exp153/Model58'
  model_path = '/data/SNUBH/SSC/exp151/Model58/Model58'
  result_serial = paste0('result-', sprintf('%03d', pre_serial))
 
  mm_train = merge_mm_df(pre_train_path, pre_serial, 1, lgs_clv_fit, df_train, 'ntrain.npy')
  mm_val = merge_mm_df(model_path, pre_serial, 3, lgs_clv_fit, df_val, 'trval.npy')
  mm_test = merge_mm_df(model_path, pre_serial, 3, lgs_clv_fit, df_test, 'test.npy')
  mm_test2 = merge_mm_df(model_path, pre_serial, 3, lgs_clv_fit, df_test2, 'test23.npy')
  
  mm_fit = glm('LABEL ~ LOGIT + NN_LOGIT', data=mm_train, family='binomial')
  print(summary(mm_fit))
  
  mm_weight = mm_fit$coefficients[-1]
  mm_bias = mm_fit$coefficients[1]
  output = list(mm_weight, mm_bias)

  json_path = paste(pre_train_path, result_serial, 'mm_pre_weight.json', sep='/')
  write_json(output, json_path)
  
  roc_dl_train = roc(mm_train$LABEL, mm_train$PROB)
  roc_dl_val = roc(mm_val$LABEL, mm_val$PROB)
  roc_dl_test = roc(mm_test$LABEL, mm_test$PROB)
  roc_dl_test2 = roc(mm_test2$LABEL, mm_test2$PROB)
  
  roc_dl_output = c(roc_dl_train$auc, roc_dl_val$auc, roc_dl_test$auc, roc_dl_test2$auc)
  print(roc_dl_output)
 
  pred_mm_train = predict(mm_fit, newdata=mm_train, type='response')
  pred_mm_val = predict(mm_fit, newdata=mm_val, type='response')
  pred_mm_test = predict(mm_fit, newdata=mm_test, type='response')
  pred_mm_test2 = predict(mm_fit, newdata=mm_test2, type='response')
  
  roc_mm_train = roc(mm_train$LABEL, pred_mm_train)  # AUC: 0.9290
  roc_mm_val = roc(mm_val$LABEL, pred_mm_val)  # AUC: 0.8232
  roc_mm_test = roc(mm_test$LABEL, pred_mm_test)  # AUC: 0.8293
  roc_mm_test2 = roc(mm_test2$LABEL, pred_mm_test2)  # AUC: 0.8260
  
  roc_mm_output = c(roc_mm_train$auc, roc_mm_val$auc, roc_mm_test$auc, roc_mm_test2$auc)
  print(roc_mm_output)
}

pre_train_mm(0, lgs_clv_fit, df_train)
pre_train_mm(3, lgs_clv_fit, df_train)
pre_train_mm(7, lgs_clv_fit, df_train)
pre_train_mm(11, lgs_clv_fit, df_train)
