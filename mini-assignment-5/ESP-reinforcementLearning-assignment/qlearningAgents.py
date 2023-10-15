# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """



  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.epsilon = args['epsilon']    # exploration rate
    self.alpha = args['alpha']        # learning rate
    self.gamma = args['gamma']        # discount factor
    self.actionFn = args['actionFn']

    self.epsilon = 0.7

    print("debug-init values")
    print(self.epsilon)
    print(self.alpha)
    print(self.gamma)

    # Store Q-values
    self.qValues = {}

  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    if (state, action) not in self.qValues:
        return 0.0
    else:
        return self.qValues[(state, action)]
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return 0.0

    # Find max Q-value among legal actions
    max_q_value = max(self.getQValue(state, action) for action in legal_actions)

    return max_q_value
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    #print("getpolicy")
    
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None

    # Get all actions with maximum Q-value and randomly choose the best
    max_q_value = self.getValue(state)
    best_actions = [action for action in legal_actions if self.getQValue(state, action) == max_q_value]
    best_action = random.choice(best_actions)

    return best_action
 
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    #print("getaction")

    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None

    # Choose a random action with probability epsilon, else use best policy action
    if util.flipCoin(self.epsilon):
        return random.choice(legal_actions)
    best_action = self.getPolicy(state)

    return best_action
 
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    #print("update")

    # Get the Q-value for current state 
    current_q_value = self.getQValue(state, action)

    # Get the maximum Q-value for next state
    next_max_q_value = self.getValue(nextState)

    # Apply the Q-Learning update rule to compute the new Q-value
    # new q = cur Q + alpha(reward for taking action + gamma*max expected reward - cur Q)
    new_q_value = current_q_value + self.alpha*(reward + self.gamma * next_max_q_value - current_q_value)
    self.qValues[(state, action)] = new_q_value