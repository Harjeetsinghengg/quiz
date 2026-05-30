import pandas as pd

class QuestionManager:
    def __init__(self, file):
        self.data = pd.read_excel(file)
        self.reset()

    def reset(self):
        self.questions = self.data.sample(n=5).reset_index(drop=True)
        self.index = 0
        self.correct = 0
        self.wrong = 0

    def get_current(self):
        return self.questions.iloc[self.index]

    def check_answer(self, selected):
        row = self.get_current()
        if selected == row["answer"]:
            self.correct += 1
        else:
            self.wrong += 1
        self.index += 1

    def finished(self):
        return self.index >= 5
