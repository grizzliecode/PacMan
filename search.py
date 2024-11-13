# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import copy

import util
from game import Directions
from typing import List


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def dfSearch(state, problem: SearchProblem, visited: set) -> (List[Directions], bool, set):
    visited.add(state)
    if problem.isGoalState(state):
        return [], True, visited
    nextMoves = problem.getSuccessors(state)
    for move in nextMoves:
        if move[0] not in visited:
            result = dfSearch(move[0], problem, visited)
            visited.union(result[2])
            if result[1]:
                return [move[1]] + result[0], True, visited
    return [], False, visited


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    result = dfSearch(problem.getStartState(), problem, set())
    return result[0]


def bfSearch(startState, problem: SearchProblem) -> List[Directions]:
    visited = set()
    visited.add(startState)
    queue = [(startState, [])]
    while len(queue) > 0:
        node = queue.pop(0)
        state = node[0]
        directions = node[1]
        if problem.isGoalState(state):
            return directions
        nextMoves = problem.getSuccessors(state)
        for move in nextMoves:
            if move[0] not in visited:
                visited.add(move[0])
                queue.append((move[0], directions + [move[1]]))
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    result = bfSearch(problem.getStartState(), problem)
    return result


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    result = []
    visited = []
    q = util.PriorityQueue()
    # visited.append(problem.getStartState())
    q.push((problem.getStartState(), []), 0)
    while not q.isEmpty():
        curr_state, actions = q.pop()
        if problem.isGoalState(curr_state):
            result = actions
            break
        if curr_state not in visited:
            visited.append(curr_state)
            for successor in problem.getSuccessors(curr_state):
                new_action = copy.deepcopy(actions) + [successor[1]]
                cost = problem.getCostOfActions(new_action)
                    # print(successor[0], problem.getCostOfActions(new_action))
                q.update((successor[0], new_action), cost)


    return result


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStar(startState, problem: SearchProblem, heuristic) -> List[Directions]:
    visited = set()
    pq = util.PriorityQueue()
    pq.push((startState, []), 0)
    while not pq.isEmpty():
        node = pq.pop()
        state = node[0]
        directions = node[1]
        if state in visited:
            continue
        visited.add(state)
        if problem.isGoalState(state):
            return directions
        nextMoves = problem.getSuccessors(state)
        for move in nextMoves:
            # if move[0] not in visited:
                nextDirections = directions + [move[1]]
                nextCost = problem.getCostOfActions(nextDirections) + heuristic(move[0], problem)
                pq.push((move[0], nextDirections), nextCost)
    return []

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return aStar(problem.getStartState(), problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
