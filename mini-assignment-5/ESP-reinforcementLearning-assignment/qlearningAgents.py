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

    self.epsilon = args['epsilon']
    self.alpha = args['alpha']
    self.gamma = args['gamma']
    self.actionFn = args['actionFn']

    # Create a dictionary to store Q-values
    self.qValues = {}

  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    # Check if the state-action pair exists in the Q-values dictionary
    if (state, action) in self.qValues:
        return self.qValues[(state, action)]
    else:
        return 0.0  # Return 0.0 for unseen state-action pairs
 
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    # Get the legal actions for the given state
    legal_actions = self.getLegalActions(state)

    # If there are no legal actions, return 0.0
    if not legal_actions:
        return 0.0

    # Find the maximum Q-value among legal actions
    max_q_value = max(self.getQValue(state, action) for action in legal_actions)

    return max_q_value
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    # Get the legal actions for the given state
    legal_actions = self.getLegalActions(state)

    # If there are no legal actions, return None (terminal state)
    if not legal_actions:
        return None

    # Find the action with the maximum Q-value
    best_action = max(legal_actions, key=lambda action: self.getQValue(state, action))

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
    # Get the legal actions for the given state
    legal_actions = self.getLegalActions(state)

    # If there are no legal actions, return None (terminal state)
    if not legal_actions:
        return None

    # Choose a random action with probability self.epsilon
    if util.flipCoin(self.epsilon):
        return random.choice(legal_actions)
    
    # Otherwise, choose the best policy action (maximum Q-value)
    best_action = max(legal_actions, key=lambda action: self.getQValue(state, action))
    
    return best_action

  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # Calculate the Q-value for the current state-action pair
    current_q_value = self.getQValue(state, action)

    # Find the maximum Q-value for the next state over all possible actions
    max_next_q_value = max(self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState))

    # Update the Q-value using the Q-learning formula
    updated_q_value = current_q_value + self.alpha * (reward + self.gamma * max_next_q_value - current_q_value)

    # Update the Q-value in the Q-values dictionary
    self.qValues[(state, action)] = updated_q_value
