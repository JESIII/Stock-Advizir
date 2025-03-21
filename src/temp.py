import data_prep
from pathlib import Path
import train
from paper_test import do_trade_nn

#data_prep.download_dataset_from_hf()

current_file_path = Path(__file__).resolve()
current_dir_path = current_file_path.parent
relative_file_path = current_dir_path.parent / "data" / "training" / "daily-stocks.parquet"
relative_dir_path = current_dir_path.parent / "data" / "output"
# data_prep.load_and_process_single_parquet_file(relative_file_path, relative_dir_path)
# nn_model, label_encoder = train.train_neural_network_classifier(relative_dir_path.joinpath("daily-stocks_processed.parquet"))

# train.save_models(nn_model, "nn_model")
# train.save_label_encoder(label_encoder, "nn_label_encoder")
# do_trade_nn("AMD")
