from distutils.log import debug
from flask import Flask, render_template, request, jsonify, redirect
import model_prediction
import runModel
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('submit.html')

    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    story = data['story']
    question = data['question']
    answer = data['answer']
    prediction = model_prediction.Prediction()
    print(f'story, question, answer:: {story} {question} {answer}')
    pred_answer = prediction.predict(story, question, answer)
    print(pred_answer)
    return jsonify({'pred_answer': pred_answer})



if __name__ == '__main__':
    app.run(debug=True)




