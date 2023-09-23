class Vocabulary:
    def __init__(self):
        self.vocab = set()
        self.max_story_len = 0
        self.max_question_len = 0
    
    def create_vocab(self, data):
        for story, question, answer in data:
            self.vocab = self.vocab.union(set(story))
            self.vocab = self.vocab.union(set(question))    

        self.vocab.add('no')
        self.vocab.add('yes')    

    def get_vocab(self):
        if len(self.vocab) != 0:
            return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def get_max_story_len(self):
        return self.max_story_len

    def get_max_question_len(self):
        return self.max_question_len    
