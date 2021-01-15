import gym
import numpy as np

from agent import Agent
from environment import ArcEnvironment
from state import State

if __name__ == '__main__':
    # initialize the arc task
    start = np.array([[0, 0], [1, 1]])
    end = np.array([[1, 1], [0, 0]])
    state = State(start, end)

    # initialize the environment
    env = ArcEnvironment(state)

    # evaluate and train and reevaluate the agent
    a = Agent(env)
    #episodes, total_epochs, total_penalties = a.evaluate(episodes=3)
    #a.train(episodes=5)
    #episodes, total_epochs, total_penalties = a.evaluate(episodes=3)




