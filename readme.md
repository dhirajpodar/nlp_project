
**Project Name**: NLP Question-Answering with Memory Networks

**Description**: This repository contains an NLP project that utilizes memory networks to predict answers from given questions and stories. The project is inspired by the research paper "End to End Memory Networks" by Facebook AI Research, New York ([Read the Paper](https://arxiv.org/pdf/1503.08895.pdf)).

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Training Details](#training-details)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project consists of stories, questions, and answers organized as a list of tuples. Each tuple contains a story, a question related to the story, and the corresponding answer.

**Example:**

**Story**
```
'Mary moved to the bathroom . Sandra journeyed to the bedroom .'
```

**Question**
```
'Is Sandra in the hallway ?'
```

**Answer**
```
'no'
```

## Model

The model used in this project is based on the architecture presented in the referenced research paper. It employs memory networks for question-answering tasks. You can find the model implementation in the `model.py` file.

![Model Architecture](/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/app/model_architecture.png)

## Training Details

The model was trained for 100 epochs using the following configurations:

- Optimizer: rmsprop
- Loss Function: categorical_crossentropy

Here is a training plot showing the evolution of training and validation loss:

![Training Plot](/Users/dhirajpoddar/Documents/Studies/project_solar/nlp_chat_bot/plot2.png)

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies `requirements.txt` file.
3. Preprocess your data using the `preprocess.py` script.
4. Train the model using the `train.py` script.
5. Run the Flask application to interact with the trained model using `main.py`.



## License

This project is licensed under the [MIT License](LICENSE).

