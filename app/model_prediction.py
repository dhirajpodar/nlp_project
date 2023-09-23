from distutils.command.config import config
import tensorflow as tf
import preprocess
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import preprocess
import pickle
import numpy as np



def load_model():
    model = tf.keras.models.load_model('/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/saved_model/chatbot_10.h5')
    return model

class Prediction:
    def __init__(self):
        self.vocab = set()
        self.max_story_len = 0
        self.max_question_len = 0
        self.model = load_model()
        self.tokenizer = Tokenizer(filters=[])

    def prepare_token(self):
        with open('/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/data/train_qa.txt', 'rb') as f:
            train_data = pickle.load(f)

        with open('/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/data/test_qa.txt','rb') as f:
            test_data = pickle.load(f)   

        _, (story_test, question_test, answer_test) , self.vocab = preprocess.load_and_preprocess(train_data, test_data)
        self.max_story_len = self.vocab.get_max_story_len
        self.max_question_len = self.vocab.get_max_question_len
        self.tokenizer.fit_on_texts(self.vocab.get_vocab())
        return test_data

    def predict(self, story, question, answer):
        

        test_data = self.prepare_token() 
        s = test_data[0][0]
        q = test_data[0][1]
        a = test_data[0][2]

        # print(test_data)
        t_story, t_question, _ = preprocess.vectorize_stories([(s,q,a)], self.tokenizer.word_index, self.vocab.get_max_story_len(), self.vocab.get_max_question_len())
        
        if self.model is not None:
            pred_result = self.model.predict([t_story, t_question])
            val_max = np.argmax(pred_result)
            for key, val in self.tokenizer.word_index.items():
                if val == val_max:    
                    return key
        else:
            print('issue with the model. model cannot be loaded.')            
    

        return 'error'