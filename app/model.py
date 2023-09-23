from keras.models import Sequential, Model
from keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM



def create_model(max_story_len, max_question_len, vocab_size):
    input_sequence = Input((max_story_len,))
    question = Input((max_question_len,))


    # INPUT ENCODER M
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
    input_encoder_m.add(Dropout(0.4))

    #(samples, story_maxlen, embedding_dim)

    # INPUT ENCODER C
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
    input_encoder_c.add(Dropout(0.4))

    #OUTPUT
    #(samples, story_maxlen, embedding_dim)

    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len))
    question_encoder.add(Dropout(0.4))

    #(Samples, query_maxlen, embedding_dim)

    # ENCODED <---- ENCODER(INPUT)

    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    match = dot([input_encoded_m,question_encoded], axes=(2,2))
    match = Activation('softmax')(match)

    response = add([match, input_encoded_c])
    response = Permute((2,1))(response)

    answer = concatenate([response,question_encoded])

    answer = LSTM(32)(answer)
    answer=Dropout(0.5)(answer)

    answer = Dense(vocab_size)(answer) # (samples, vocab_size) # YES/NO 000

    answer = Activation('softmax')(answer)

    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model