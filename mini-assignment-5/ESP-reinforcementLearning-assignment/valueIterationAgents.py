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
     
    for _ in range(iterations):
        newValues = util.Counter()  # Create a new Counter for updated values
        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                # Calculate Q-value for each possible action
                action_values = []
                for action in mdp.getPossibleActions(state):
                    q_value = sum(
                        prob * (mdp.getReward(state, action, next_state) + discount * self.values[next_state])
                        for next_state, prob in mdp.getTransitionStatesAndProbs(state, action)
                    )
                    action_values.append(q_value)
                # Update the value of the state to the maximum Q-value
                newValues[state] = max(action_values)
        self.values = newValues  # Update values with the newly calculated values

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
    # Check if the state is a terminal state; Q-value is 0 for terminal states
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
    # Check if the state is a terminal state; return None if it is
    if self.mdp.isTerminal(state):
        return None

    # Get the legal actions for the given state
    legal_actions = self.mdp.getPossibleActions(state)

    if not legal_actions:
        return None  # No legal actions, return None

    # Filter actions with the maximum Q-value and randomly choose best if multiple are good
    max_q_value = self.getValue(state)
    best_actions = [action for action in legal_actions if self.getQValue(state, action) == max_q_value]
    best_action = random.choice(best_actions)

    return best_action

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)