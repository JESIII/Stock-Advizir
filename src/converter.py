import onnx
import pickle
from onnxmltools.convert import convert_xgboost

from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

# Load your XGBoost model
model = pickle.load(open('./models/xgboost_model.pkl', 'rb'))

# Rename the feature columns to match the pattern 'f%d'
X = all_data[['rsi', 'trix', 'stoch', 'ppo', 'vzo']]
X.columns = [f'f{index}' for index in range(X.shape[1])]

onnx_model_converted = convert_xgboost(model, 'tree-based classifier',
                             [('input', FloatTensorType([None, 5]))],
                             target_opset=15)

onnx.save_model(onnx_model_converted, "./models/xgboost_model_onnx.onnx")