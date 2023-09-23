import tensorflow as tf


def train(data, model, batch_size, epoch):
    

    (story_train, question_train, answer_train) = data['train']
    (story_test, question_test, answer_test) = data['test']

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = '/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/models/bot_best_model.h5',
        save_best_only = True,
        monitor='val_loss',
        mode= 'min',
        verbose = 1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=25
    )
    history = model.fit([story_train, question_train],answer_train, 
                    batch_size, 
                    epoch, 
                    validation_data=([story_test,question_test], answer_test),
                    callbacks=[checkpoint_callback, early_stopping_callback])

    losses = {'loss': history.history['loss'],
            'val_loss':history.history['val_loss']}    
    # print(f"acc:{history.history['accuracy']}")
    print('Accuracy :: {:.2f}%'.format(history.history['accuracy'][0] * 100))
    return losses        