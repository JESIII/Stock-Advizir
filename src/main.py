import data_prep
import train

output_dir = './data/output/'
# Process the datasets
# data_prep.load_process_and_save_data(output_dir)

# # Train and compare classifiers
# xgb_model, label_encoder = train.train_xgboost_classifier(output_dir)

# train.save_models(xgb_model, "xgboost_model")
# train.save_label_encoder(label_encoder, "xgboost_label_encoder")

# train.plot_test_data()
# train.plot_test_data('HistoricalData_1738966623767_processed.csv')