# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import random
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
  
    #print(iterations)

    iteration = 0
    while iteration < iterations:
        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                action_values = []
                for action in mdp.getPossibleActions(state):
                    action_values.append(self.getQValue(state,action))
                self.values[state] = max(action_values)
        iteration += 1
        #print(self.values)

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    if self.mdp.isTerminal(state):
        return 0.0

    # Calculate the Q-value using the Bellman equation
    q_value = sum(
        prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
    )

    return q_value

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    if self.mdp.isTerminal(state):
        return None
    
    legal_actions = self.mdp.getPossibleActions(state)
    if not legal_actions:
        return None  # No legal actions, return None

    # Filter actions with the maximum Q-value and randomly choose best if multiple are good
    max_q_value = self.getValue(state)
    #print("valid actions: " + str(legal_actions))
    #print("max_q_value: " + str(max_q_value))
    best_actions = [action for action in legal_actions if self.getQValue(state, action) == max_q_value]
    if best_actions:
      best_action = random.choice(best_actions)
      return best_action
    else:
        return random.choice(legal_actions)  # Choose a random valid action if there are no best actions

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)