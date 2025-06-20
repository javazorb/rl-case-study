class BaseAgent:
    def train(self, train_set, val_set):
        raise NotImplementedError

    def evaluate(self, environment):
        raise NotImplementedError