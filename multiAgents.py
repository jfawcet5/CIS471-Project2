# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"   
        fdist = 99999 # Initial closest food distance
        M = newFood.height
        N = newFood.width
        for i in range(N):
            for j in range(M):
                if newFood[i][j]:
                    d = manhattanDistance(newPos, (i, j))
                    if d < fdist:
                        fdist = d # new closest food distance

        f1 = 1 / (fdist + 1) # Larger value of f1 for closer food

        f2 = max(newScaredTimes) # Not entirely sure what newScaredTimes is, but Im taking the largest one

        # List of adjacent positions to pacman
        adjacentPos = [(newPos[0] - 1, newPos[1]), (newPos[0] + 1, newPos[1]), (newPos[0], newPos[1] - 1), (newPos[0], newPos[1] + 1)]

        f3 = 1 # f3 is related to how close the ghosts are to pacman
        for ghost in newGhostStates:
            gpos = ghost.getPosition()
            if (newPos == gpos): # If pacman's position overlaps a ghosts position
                f3 = 0 # f3 is 0 because pacman on same position as ghost = lose
                break;
            for pos in adjacentPos:
                if gpos == pos: # If ghost is adjacent to pacman
                    f3 = 0 # f3 is 0 because we do not want to be directly next to ghost
                    break;

        if (f3 == 0): # if f3 = 0, there is a very high chance of losing ==> return low value
            return 0

        return successorGameState.getScore() + .1*f1 + .3*f2 + .6*f3 # return weighted sum of features

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.cur_depth = 0
        self.numAgents = gameState.getNumAgents()
        self.stopPoint = (self.numAgents * self.depth)

        (maxV, bestAction) = (float('-inf'), None)

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            curval = self.value(successor, 1, 1)
            if curval > maxV:
                (maxV, bestAction) = curval, action

        return bestAction

    def value(self, gameState, curAgent, curDepth):
        if gameState.isWin() or gameState.isLose() or curDepth == self.stopPoint:
            return self.evaluationFunction(gameState)

        if curAgent == 0:
            return self.max_value(gameState, curAgent, curDepth)
        else:
            return self.min_value(gameState, curAgent, curDepth)

    def max_value(self, gameState, curAgent, curDepth):
        v = float('-inf')
        for action in gameState.getLegalActions(curAgent):
            successor = gameState.generateSuccessor(curAgent, action)
            v = max(v, self.value(successor, (curAgent + 1) % self.numAgents, curDepth + 1))
        return v

    def min_value(self, gameState, curAgent, curDepth):
        v = float('inf')
        for action in gameState.getLegalActions(curAgent):
            successor = gameState.generateSuccessor(curAgent, action)
            v = min(v, self.value(successor, (curAgent + 1) % self.numAgents, curDepth + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaSearch(gameState)

    def alphaBetaSearch(self, gameState):
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        act = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tmp = self.min_value(successor, 1, 0, alpha, beta)
            if v < tmp:
                v = tmp
                act = action

            if v > beta:
                return v
            alpha = max(alpha, tmp)
        return act

    def max_value(self, gameState, curDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth:
            return self.evaluationFunction(gameState)
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.min_value(successor, 1, curDepth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, curAgent, curDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth:
            return self.evaluationFunction(gameState)
        v = float('inf')
        for action in gameState.getLegalActions(curAgent):
            successor = gameState.generateSuccessor(curAgent, action)
            if curAgent == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(successor, curDepth + 1, alpha, beta))
            else:
                v = min(v, self.min_value(successor, curAgent + 1, curDepth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        self.numAgents = gameState.getNumAgents() # Number of agents, including pacman and all ghosts

        self.stopPoint = (self.numAgents * self.depth) # How many calls to self.value() before reaching maximum depth

        maxV, bestMove = -9999999, None # Current maximum expected utility and best move so far

        legalMoves = gameState.getLegalActions(0) # Available moves at the beginning state

        for move in legalMoves: # Iterate through available moves

            curState = gameState.generateSuccessor(0, move) # generate new state based on current move
            
            curVal = self.value(curState, 1, 1) # Run expectimax search on current state to get maximum expected value

            if curVal > maxV: # If the calculated expected utility for the current state is higher than 'maxV'

                maxV, bestMove = curVal, move # Save new values of 'maxV' and 'bestMove'

        return bestMove # return the move with the highest expected utility

    def value(self, gameState, curAgent, curdepth):
        """ value() function based on pseudocode from lecture slides. Makes recursive 
            calls to self.max_value() and self.exp_value() until the stopping point
            (maximum depth) has been reached. 

            'gameState' is the current game state
            'curAgent' is the current agent that is taking action (0 = pacman, 1+ = ghost)
            'curdepth' records the depth of the recursive calls to limit depth
        """
        if gameState.isWin() or gameState.isLose() or (curdepth == self.stopPoint):
            return self.evaluationFunction(gameState)

        if curAgent == 0:
            return self.max_value(gameState, curAgent, curdepth)
        else:
            return self.exp_value(gameState, curAgent, curdepth)

    def max_value(self, gameState, curAgent, curdepth):
        """ max_value() function based on pseudocode from lecture slides. Returns
            the maximum value of the current gameState based on recursive calls to 
            self.value()

            'gameState' is the current game state
            'curAgent' is the current agent that is taking action (0 = pacman, 1+ = ghost)
            'curdepth' records the depth of the recursive calls to limit depth
        """
        v = -999999 # Initial highest expected utility 

        legalMoves = gameState.getLegalActions(curAgent) # Available moves to current agent

        for move in legalMoves: # Iterate through legal moves

            newState = gameState.generateSuccessor(curAgent, move) # Generate new state based on current move/action

            newV = self.value(newState, (curAgent + 1) % self.numAgents, curdepth + 1) # Calculate maximum expected value for 'newState'

            if newV > v: # If new expected value is higher than max

                v = newV # Save new expected value as max

        return v # return maximum expected value 

    def exp_value(self, gameState, curAgent, curdepth):
        """ exp_value() function based on pseudocode from lecture slides. Returns
            the expected value of the current gameState based on recursive calls to 
            self.value()

            'gameState' is the current game state
            'curAgent' is the current agent that is taking action (0 = pacman, 1+ = ghost)
            'curdepth' records the depth of the recursive calls to limit depth
        """
        v = 0 # Initial expected utility

        legalMoves = gameState.getLegalActions(curAgent) # Available moves to current agent

        p = 1 / (len(legalMoves)) # Uniform probability for each legal move

        for move in legalMoves: # Iterate through legal moves

            newState = gameState.generateSuccessor(curAgent, move) # Generate new state based on current move/action

            v += p * self.value(newState, (curAgent + 1) % self.numAgents, curdepth + 1) # Add (probability * calculated expected utility) to v

        return v # Return expected utility

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
