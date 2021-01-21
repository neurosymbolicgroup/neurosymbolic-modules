import gym


class ArcEnvironment(gym.Env):
    """
    Reward:
        Gives reward when task solved.
        Gives penalty when task timeout.
    Starting State:
        The root and leaf of the tree.
    State:
        The developing tree, and actions taken to get there.
    """

    def __init__(self, init_state):
        """
        Initialize environment using the arc task
        """

        # self.task = None
        self.state = init_state

        self.setup()

    def step(self, action):
        """
        (1) Apply the action
        (2) Update environment's state
        """
        # Apply action
        # ...

        # Update environment's state
        # ...

        # Determine reward
        # if self.state.done:
        #     reward = ...

        # return self.state, reward, self.state.done


    def setup(self):
        """
        Set up initial state of environment.
        """

        return self.state
