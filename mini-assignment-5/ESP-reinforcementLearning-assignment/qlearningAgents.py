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

    print("debug-init values")
    print(self.epsilon)
    print(self.alpha)
    print(self.gamma)

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
    # Get the legal actions for the given state, if none, return 0.0
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return 0.0

    # Find the maximum Q-value among legal actions
    max_q_value = max(self.getQValue(state, action) for action in legal_actions)

    return max_q_value
    
  def getPolicy(self, state):

    #print("getpolicy")

    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    # Get the legal actions for the given state, if it's a terminal state, return None
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None

    # Find the maximum Q-value among legal actions
    max_q_value = self.getValue(state)

    # Filter actions with the maximum Q-value and randomly choose best if multiple are good
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

    # Get the legal actions for the given state, if terminal state, return None
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None

    # Choose a random action with probability self.epsilon, else use policy
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

    # Get the current Q-value for the (state, action) pair
    current_q_value = self.getQValue(state, action)

    # Calculate the maximum Q-value for the next state
    next_max_q_value = self.getValue(nextState)

    # Apply the Q-Learning update rule to compute the new Q-value
    new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * next_max_q_value)

    # Update the Q-value for the (state, action) pair in the Q-values dictionary
    self.qValues[(state, action)] = new_q_value