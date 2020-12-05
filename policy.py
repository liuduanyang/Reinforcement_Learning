from utils import *
import gym
import numpy as np
from collections import deque

# Run this cell without modification

def evaluate(policy, num_episodes, seed=0, env_name='FrozenLake8x8-v0', render=False, gamma=1.0):
    """[TODO] You need to implement this function by yourself. It
    evaluate the given policy and return the mean episode reward.
    We use `seed` argument for testing purpose.
    You should pass the tests in the next cell.

    :param policy: a function whose input is an interger (observation)
    :param num_episodes: number of episodes you wish to run
    :param seed: an interger, used for testing.
    :param env_name: the name of the environment
    :param render: a boolean flag. If true, please call _render_helper
    function.
    :return: the averaged episode reward of the given policy.
    """

    # Create environment (according to env_name, we will use env other than 'FrozenLake8x8-v0')
    env = gym.make(env_name)

    # Seed the environment
    env.seed(seed)

    # Build inner loop to run.
    # For each episode, do not set the limit.
    # Only terminate episode (reset environment) when done = True.
    # The episode reward is the sum of all rewards happen within one episode.
    # Call the helper function `render(env)` to render
    rewards = []
    for i in range(num_episodes):
        # reset the environment
        obs = env.reset()
        # act = policy(obs)
        step_idx = 0
        ep_reward = 0
        while True:
            # [TODO] run the environment and terminate it if done, collect the
            # reward at each step and sum them to the episode reward.
            if render:
                _render_helper(env)
            obs, reward, done, info = env.step(int(policy(obs)))
            ep_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        
        rewards.append(ep_reward)

    return np.mean(rewards)

class TabularRLTrainerAbstract:
    """This is the abstract class for tabular RL trainer. We will inherent the specify 
    algorithm's trainer from this abstract class, so that we can reuse the codes like
    getting the dynamic of the environment (self._get_transitions()) or rendering the
    learned policy (self.render())."""
    
    def __init__(self, env_name='FrozenLake8x8-v0', model_based=True):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.n
        
        self.model_based = model_based

    def _get_transitions(self, state, act):
        """Query the environment to get the transition probability,
        reward, the next state, and done given a pair of state and action.
        We implement this function for you. But you need to know the 
        return format of this function.
        """
        self._check_env_name()
        assert self.model_based, "You should not use _get_transitions in " \
            "model-free algorithm!"
        
        # call the internal attribute of the environments.
        # `transitions` is a list contain all possible next states and the 
        # probability, reward, and termination indicater corresponding to it
        transitions = self.env.env.P[state][act]

        # Given a certain state and action pair, it is possible
        # to find there exist multiple transitions, since the 
        # environment is not deterministic.
        # You need to know the return format of this function: a list of dicts  返回字典列表
        ret = []
        for prob, next_state, reward, done in transitions:
            ret.append({
                "prob": prob,
                "next_state": next_state,
                "reward": reward,
                "done": done
            })
        return ret
    
    def _check_env_name(self):
        assert self.env_name.startswith('FrozenLake')

    def print_table(self):
        """print beautiful table, only work for FrozenLake8X8-v0 env. We 
        write this function for you."""
        self._check_env_name()
        print_table(self.table)

    def train(self):
        """Conduct one iteration of learning."""
        raise NotImplementedError("You need to override the "
                                  "Trainer.train() function.")

    def evaluate(self):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 1000 episodes when seed=0."""
        result = evaluate(self.policy, 1000, env_name=self.env_name)
        return result

    def render(self):
        """Reuse your evaluate function, render current policy 
        for one episode when seed=0"""
        evaluate(self.policy, 1, render=True, env_name=self.env_name)


class PolicyItertaionTrainer(TabularRLTrainerAbstract):
    def __init__(self, gamma=1.0, eps=1e-10, env_name='FrozenLake8x8-v0'):
        super(PolicyItertaionTrainer, self).__init__(env_name)

        # discount factor
        self.gamma = gamma

        # value function convergence criterion
        self.eps = eps

        # build the value table for each possible observation
        self.table = np.zeros((self.obs_dim,))

        # [TODO] you need to implement a random policy at the beginning.
        # It is a function that take an integer (state or say observation)
        # as input and return an interger (action).
        # remember, you can use self.action_dim to get the dimension (range)
        # of the action, which is an integer in range
        # [0, ..., self.action_dim - 1]
        # hint: generating random action at each call of policy may lead to
        #  failure of convergence, try generate random actions at initializtion
        #  and fix it during the training.
        self.policy_table = np.random.choice(self.action_dim, size=(self.obs_dim))
        # self.policy = self.obs_to_policy
        self.policy = lambda obs: self.policy_table[obs]

        # test your random policy
        # test_random_policy(self.policy, self.env)
        
        
    def obs_to_policy(self,obs):
        return self.policy_table[obs]

    def train(self):
        """Conduct one iteration of learning."""
        # [TODO] value function may be need to be reset to zeros.
        # if you think it should, than do it. If not, then move on.
        # hint: the value function is equivalent to self.table,
        #  a numpy array with length 64.
        self.table = np.zeros((self.obs_dim,))

        self.update_value_function()
        self.update_policy()

    def update_value_function(self):
        count = 0  # count the steps of value updates
        should_break = False
        while True:
            old_table = self.table.copy()

            for state in range(self.obs_dim):
                act = self.policy(state)
                transition_list = self._get_transitions(state, act)
                
                state_value = 0
                for transition in transition_list:
                    prob = transition['prob']
                    reward = transition['reward']
                    next_state = transition['next_state']
                    done = transition['done']
                    
                    # [TODO] what is the right state value?
                    # hint: you should use reward, self.gamma, old_table, prob,
                    # and next_state to compute the state value
                    state_value += prob * (reward + self.gamma * old_table[next_state])

                # update the state value
                self.table[state] = state_value

            # [TODO] Compare the old_table and current table to
            #  decide whether to break the value update process.
            # hint: you should use self.eps, old_table and self.table
            if(np.sum(np.abs(old_table-self.table))<=self.eps):
                should_break = True

            if should_break:
                break
            count += 1
            if count % 200 == 0:
                # disable this part if you think debug message annoying.
                print("[DEBUG]\tUpdated values for {} steps. "
                      "Difference between new and old table is: {}".format(
                    count, np.sum(np.abs(old_table - self.table))
                ))
            if count > 4000:
                print("[HINT] Are you sure your codes is OK? It shouldn't be "
                      "so hard to update the value function. You already "
                      "use {} steps to update value function within "
                      "single iteration.".format(count))
            if count > 6000:
                raise ValueError("Clearly your code has problem. Check it!")

    def update_policy(self):
        """You need to define a new policy function, given current
        value function. The best action for a given state is the one that
        has greatest expected return.

        To optimize computing efficiency, we introduce a policy table,
        which take state as index and return the action given a state.
        """
        # policy_table = np.zeros([self.obs_dim, ], dtype=np.int)

        for state in range(self.obs_dim):
            state_action_values = [0] * self.action_dim

            # [TODO] assign the action with greatest "value"
            # to policy_table[state]
            # hint: what is the proper "value" here?
            #  you should use table, gamma, reward, prob,
            #  next_state and self._get_transitions() function
            #  as what we done at self.update_value_function()
            #  Bellman equation may help.
            for action in range(self.action_dim):
                transition_list = self._get_transitions(state, action)
                for transition in transition_list:
                    prob = transition['prob']
                    next_state = transition['next_state']
                    reward = transition['reward']
                    done = transition['done']
                    
                    state_action_values[action] += (prob * (reward + self.gamma * self.table[next_state]))
            
            best_action = np.argmax(state_action_values)
            
            self.policy_table[state] = best_action


# Solve the TODOs and remove `pass`

# Managing configurations of your experiments is important for your research.
default_pi_config = dict(
    max_iteration=1000,
    evaluate_interval=1,
    gamma=1.0,
    eps=1e-10
)


def policy_iteration(train_config=None):
    config = default_pi_config.copy()
    if train_config is not None:
        config.update(train_config)
        
    trainer = PolicyItertaionTrainer(gamma=config['gamma'], eps=config['eps'])

    old_policy_result = {
        obs: -1 for obs in range(trainer.obs_dim)
    }

    should_stop = False
    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line
        
        # [TODO] compare the new policy with old policy to check whether
        #  should we stop. If new and old policy have same output given any
        #  observation, them we consider the algorithm is converged and
        #  should be stopped.
        if(np.all(old_policy_result==trainer.policy_table)):
            should_stop = True
        
        if should_stop:
            print("We found policy is not changed anymore at "
                  "itertaion {}. Current mean episode reward "
                  "is {}. Stop training.".format(i, trainer.evaluate()))
            break
        old_policy_result = trainer.policy_table.copy()

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

            if i > 20:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer


# Run this cell without modification

# It may be confusing to call a trainer agent. But that's what we normally do.
pi_agent = policy_iteration()