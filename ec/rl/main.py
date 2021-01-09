import gym
import numpy as np

from agent import Agent
from environment import ArcEnvironment
from tasktree import TaskTree

if __name__ == '__main__':
    # initialize the arc task
    start = np.array([[0, 0], [1, 1]])
    end = np.array([[1, 1], [0, 0]])
    tasktree = TaskTree(start, end)

    # initialize the environment
    env = ArcEnvironment(tasktree)

    # evaluate and train and reevaluate the agent
    a = Agent(env)
    #episodes, total_epochs, total_penalties = a.evaluate(episodes=3)
    #a.train(episodes=5)
    #episodes, total_epochs, total_penalties = a.evaluate(episodes=3)




