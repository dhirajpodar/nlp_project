import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import preprocess
import pickle
import numpy as np
import model_prediction


prediction = model_prediction.Prediction()

story = ''
question = ''
answer = ''
result_ans = prediction.predict(story, question, answer)
print(result_ans)
