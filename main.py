# Functions
from functions import time_ms, date, get_dataset, display_data_info, plot_histograms, analyse_data_correlations, \
    separe_predictors_labels

# ====== START ======
start_time = time_ms()
print('*** Staring process ***')
print(date(), '\n')

# ====== DATA ======
print('*** DATA ***', '\n')
print('Getting the dataset...', end=' ')
train_set, test_set = get_dataset()
print('done!', '\n')

# ====== ANALYSE ======
print('*** ANALISE ***', '\n')
display_data_info(train_set)
plot_histograms(train_set)
analyse_data_correlations(train_set)

# ====== PREPARE ======
print('*** PREPARE ***', '\n')
train_predictors, train_labels = separe_predictors_labels(train_set)

# @todo continue handle missing features
