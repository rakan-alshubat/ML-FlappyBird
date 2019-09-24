import random
import cPickle as pickle
import numpy as np
from evostra import EvolutionStrategy
from ple import PLE
from ple.games.flappybird import FlappyBird
from model import Model
import sys

import numpy as np

class Model(object):

    def __init__(self):
        self.weights = [np.random.randn(8, 500), np.random.randn(500, 2), np.random.randn(1, 500)]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = np.dot(out, self.weights[0]) + self.weights[-1]
        out = np.dot(out, self.weights[1])
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
class Agent:
    GENERATION = 0;
    AGENT_HISTORY_LENGTH = 1
    NUM_OF_ACTIONS = 2
    POPULATION_SIZE = 15
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    INITIAL_EXPLORATION = 0.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 100000

    def __init__(self):
        self.model = Model()
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA,
                                    self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        x = np.argmax(prediction)
        return 119 if x == 1 else None

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def get_observation(self):
        state = self.env.getGameState()
        return np.array(state.values())

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def play(self, episodes):
        self.env.display_screen = True
        self.model.set_weights(self.es.weights)
        for episode in xrange(episodes):
            self.env.reset_game()
            observation = self.get_observation()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            score = 0

            while not done:
                action = self.get_predicted_action(sequence)
                reward = self.env.act(action)
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)
                done = self.env.game_over()
                if self.game.getScore() > score:
                    score = self.game.getScore()
            self.GENERATION = self.GENERATION + 1

            print self.GENERATION
            print "score: %d" % score
        self.env.display_screen = False

    def train(self, iterations):
        self.es.run(iterations, print_step=10)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in xrange(self.EPS_AVG):
            self.env.reset_game()
            observation = self.get_observation()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION,
                                       self.exploration - self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = random.choice([119, None])
                else:
                    action = self.get_predicted_action(sequence)

                reward = self.env.act(action)
                reward += random.choice([0.0001, -0.0001])
                total_reward += reward
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)
                done = self.env.game_over()

        return total_reward / self.EPS_AVG

#Declare Agent
MLAgent = Agent()

#Get Argument Count

if (len(sys.argv) >= 3):
    if(sys.argv[1] == "train"):
        MLAgent.train(int(sys.argv[2]))
        if(len(sys.argv) == 4):
            MLAgent.save(sys.argv[3])
        else:
            MLAgent.save()
    else:
        if(len(sys.argv) == 4):
            MLAgent.load(sys.argv[3])
        else:
            MLAgent.load()
        MLAgent.play(int(sys.argv[2]))


