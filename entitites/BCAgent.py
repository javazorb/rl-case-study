from entitites.BaseAgent import BaseAgent
from models.bc_model import BehavioralModel
import config


class BCAgent(BaseAgent):
    def __init__(self, optimizer, criterion):
        super().__init__(optimizer, criterion)
        self.device = config.get_device()
        self.model = BehavioralModel().to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_set, val_set):
        self.model.train()
        pass

    def evaluate(self, environment):
        self.model.eval()
        pass