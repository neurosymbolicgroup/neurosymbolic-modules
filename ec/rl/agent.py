

class Agent:

    def __init__(self, env):
        self.env = env

    def train(self, episodes=5, alpha=0.1, gamma=0.6, epsilon=0.1):
        pass

        # env = self.env
        # qtable = self.qtable

        # # Run episodes
        # for _ in range(episodes):
        #     state = env.reset()
        #     epochs, penalties, reward, = 0, 0, 0
            
        #     while not env.state.done:
        #         pass
         
        #         if reward < 0:
        #             penalties += 1

        #         state = next_state
        #         epochs += 1

    def evaluate(self, episodes=5, empty_q_table=False):
        pass
