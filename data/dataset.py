import numpy as np
import random
from typing import Tuple, Any
import d4rl
import gym

class EnvDataset:

    def __init__(self, env_name: str, q_learning: bool, batch_size:int) -> None:
        self.env_name = env_name
        self.q_learning = q_learning
        self.batch_size = batch_size

        self.env = gym.make(env_name)

        if self.q_learning:
            self.data = d4rl.qlearning_dataset(self.env)
        else:
            self.data = self.env.get_dataset()
        

    def sample(self, k: int | None = None) -> Tuple[Any,...]:

        if k is None:
            k = self.batch_size 
        indices = np.random.randint(0, len(self.data), k)

        states = self.data['observations'][indices]
        actions = self.data['actions'][indices]
        rewards = self.data['rewards'][indices]
        
        
        if self.q_learning:
            next_states = self.data['next_observations'][indices]
            terminals = self.data['terminals'][indices]
            return states, actions, rewards, next_states, terminals
        else:
            terminals = self.data['terminals'][indices]
            timeouts = self.data['timeouts'][indices] 
            return states, actions, rewards, timeouts, terminals



