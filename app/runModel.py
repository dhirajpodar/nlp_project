import pickle
import train
import preprocess
import model
import matplotlib.pyplot as plt

training = False

def runModel():
    with open('/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/data/train_qa.txt', 'rb') as f:
        train_data = pickle.load(f)

    with open('/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/data/test_qa.txt','rb') as f:
        test_data = pickle.load(f) 

    (story_train, question_train, answer_train), (story_test, question_test, answer_test) , vocabulary = preprocess.load_and_preprocess(train_data, test_data)

    data = { 
        'train' : (story_train, question_train, answer_train),
        'test': (story_test, question_test, answer_test)
    }

    
    _model = model.create_model(vocabulary.max_story_len, vocabulary.max_question_len, vocabulary.get_vocab_size() + 1)
    
    losses = train.train(data, _model, batch_size=32, epoch=200)
    loss, val_loss = losses['loss'], losses['val_loss']
    plot_graph(loss, val_loss)


def plot_graph(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.savefig('plot2.png')       

if training:
    runModel()

