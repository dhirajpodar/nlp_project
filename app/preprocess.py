import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import helper

tokenizer = Tokenizer(filters=[])
max_story_len = 0
max_question_len = 0

vocabulary = helper.Vocabulary()

def load_and_preprocess(train_data, test_data):
    
    all_data = train_data + test_data 
    
    vocabulary.create_vocab(all_data)
    
    vocab = vocabulary.get_vocab()
    print(vocab)
    # vocab_len = len(vocab) + 1

    # longest story, question

    vocabulary.max_story_len = max(len(data[0]) for data in all_data)
    vocabulary.max_question_len = max(len(data[1]) for data in all_data)


    train_story_text = []
    train_question_text = []
    train_answer = []

    for story, question, answer in train_data:
        train_story_text.append(story)
        train_question_text.append(question)
        train_answer.append(answer)

    
    tokenizer.fit_on_texts(vocab)    
    # train_story_seq = tokenizer.texts_to_sequences(train_story_text)
    word_index = tokenizer.word_index
    story_train, question_train, answers_train = vectorize_stories(train_data, word_index, vocabulary.max_story_len, vocabulary.max_question_len)
    story_test, question_test, answers_test = vectorize_stories(test_data, word_index, vocabulary.max_story_len, vocabulary.max_question_len)

    return (story_train, question_train, answers_train), (story_test, question_test, answers_test), vocabulary




def vectorize_stories(data, word_index, max_story_len, max_question_len):
    # train stories
    X = []

    #questions
    Xq = []

    # answer
    Y = []
  

    for story, question, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)


    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq,maxlen=max_question_len), np.array(Y))    