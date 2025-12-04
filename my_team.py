# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import random
import time
import math

import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


########################
# Global map variables #
########################

# All cells that are part of “tunnel” structures
MAP_TUNNELS = []

# Tunnels restricted to our own half (used on defense)
HOME_TUNNELS = []

# All wall locations in current layout
MAP_WALLS = []

# All legal cells that are *not* in tunnels
OPEN_ROADS = []

# All legal (non-wall) cells
FREE_CELLS = []


###########################
# Grid / graph utilities  #
###########################


def count_neighbors(cell, walkable_cells):
    """
    Count the number of legal 4-neighbors of a cell (up, down, left, right).

    Graph-theory viewpoint:
      - We treat the grid as an undirected graph.
      - The degree of a node v in this graph is:
            deg(v) = number of walkable neighbors of v.
    """
    x, y = cell
    neighbor_count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (x + dx, y + dy) in walkable_cells:
            neighbor_count += 1
    return neighbor_count


def list_neighbors(cell, walkable_cells):
    """
    Return the list of 4-neighbor cells that are in walkable_cells.
    
    """
    x, y = cell
    neighbors = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (x + dx, y + dy)
        if nxt in walkable_cells:
            neighbors.append(nxt)
    return neighbors


def grow_tunnel_layer(all_legal, current_tunnels):
    """
    Given current_tunnels, add one “layer” of new tunnel cells.

    Tunnel classification rule:
      - Let deg_legal(v) be the degree of v among all legal cells.
      - Let deg_tunnel(v) be the degree of v among cells already marked as tunnels.
      - A cell v is considered a tunnel cell if:

            deg_legal(v) - deg_tunnel(v) = 1

        i.e., v has exactly one neighbor that is *not* part of the tunnel graph.
    """
    tunnel_set = set(current_tunnels)
    grown = list(current_tunnels)

    for cell in all_legal:
        if cell in tunnel_set:
            continue

        total_deg = count_neighbors(cell, all_legal)
        tunnel_deg = count_neighbors(cell, current_tunnels)

        # exactly one neighbor that is not part of the tunnel graph
        if total_deg - tunnel_deg == 1:
            grown.append(cell)

    return grown


def compute_all_tunnels(all_legal):
    """
    Iteratively grow tunnels until no new cells are added.

    An iterative fixpoint computation:
      - Start with T_0 = empty set of tunnel cells.
      - Repeatedly apply grow_tunnel_layer to get T_{k+1}.
      - Stop when T_{k+1} = T_k.

    """
    tunnels = []
    while True:
        extended = grow_tunnel_layer(all_legal, tunnels)
        if len(extended) == len(tunnels):
            return extended
        tunnels = extended


def move_in_direction(position, direction):
    """
    Apply a movement direction to a grid position.
    """
    x, y = position
    if direction == Directions.NORTH:
        return x, y + 1
    if direction == Directions.SOUTH:
        return x, y - 1
    if direction == Directions.EAST:
        return x + 1, y
    if direction == Directions.WEST:
        return x - 1, y
    return position


def manhattan_distance(p1, p2):
    """
    Manhattan (L1) distance between two points.

    Heuristic search:
      - For positions p1 = (x1, y1), p2 = (x2, y2), we define:

            h(p1, p2) = |x1 - x2| + |y1 - y2|
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def tunnel_exit_neighbor(cell, tunnel_cells, legal_cells):
    """
    For a tunnel cell, return a neighboring cell that is not in tunnel_cells.

    This locates a “boundary” node between the tunnel component and the
    surrounding free space (used to find tunnel entries/exits).
    """
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (x + dx, y + dy)
        if nxt in legal_cells and nxt not in tunnel_cells:
            return nxt
    return None


def tunnel_component(start_cell, tunnel_cells):
    """
    From a single tunnel cell, collect the entire connected tunnel component.

    Standard Connected Components algorithm using BFS:

      - Treat tunnel_cells as nodes in a graph.
      - Use a FIFO queue to explore neighbors:
            1) Push start_cell into the queue.
            2) While queue not empty:
                   pop a cell u
                   for each neighbor v in tunnel_cells:
                       if v not visited: push v

      - The set of visited nodes is the connected component of start_cell.
    """
    if start_cell not in tunnel_cells:
        return None

    tunnel_set = set(tunnel_cells)
    visited = set()
    queue = util.Queue()
    queue.push(start_cell)

    while not queue.is_empty():
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)

        for nxt in list_neighbors(current, tunnel_set):
            if nxt not in visited:
                queue.push(nxt)

    return list(visited)


def tunnel_entry_for(cell, tunnel_cells, legal_cells):
    """
    If cell is in a tunnel, return an “entry” cell where the tunnel meets open road.

    Conceptually:
      - Compute the connected component C of 'cell' inside tunnel_cells.
      - For each t in C, if there is a neighbor n in legal_cells \ tunnel_cells,
        then n is a valid tunnel entry/exit cell.
    """
    if cell not in tunnel_cells:
        return None

    comp = tunnel_component(cell, tunnel_cells)
    if comp is None:
        return None

    for t_cell in comp:
        entry = tunnel_exit_neighbor(t_cell, tunnel_cells, legal_cells)
        if entry is not None:
            return entry
    return None


##########################
# MCTS Search Structures #
##########################


class SearchNode:
    """
    MCTS node with payload:
      (game_state, accumulated_reward, visit_count)

    In the notation from the MCTS / UCT:
      - state  ≈ s
      - accumulated_reward / visits ≈ Q(s, a) estimate for this branch
      - visit_count ≈ N(s, a)
    """

    def __init__(self, payload, node_id=0):
        state, reward_sum, visits = payload
        self.id = node_id
        self.children = []
        self.value = (state, float(reward_sum), float(visits))
        self.is_leaf = True

    def add_child(self, child_node):
        self.children.append(child_node)

    def select_child_ucb(self):
        """
        Select the child with the highest UCB1 score.

        UCB(child) = Q̄(child) + c * sqrt( ln N(parent) / N(child) )

        where:
          - Q̄(child)    = average reward for this child
          - N(parent)   = total number of visits to the parent node
          - N(child)    = number of visits to the child
          - c           = exploration constant (here ≈ 1.96)

        In code:
          - total_reward ≈ sum of rewards for this child
          - child_visits = N(child)
          - parent_visits = N(parent)
          - Q̄(child) = total_reward / child_visits
          - We fold Q̄ and child_visits into a single score stored in value.
        """
        _, _, parent_visits = self.value
        best_child = None
        best_score = float('-inf')

        for child in self.children:
            _, total_reward, child_visits = child.value
            if child_visits == 0:
                # force exploration of unvisited nodes
                return child
            ucb = total_reward + 1.96 * math.sqrt(math.log(parent_visits) / child_visits)
            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def find_parent_of(self, target_node):
        """
        Depth-first search to find parent of 'target_node'.
        """
        for child in self.children:
            if child is target_node:
                return self
            parent = child.find_parent_of(target_node)
            if parent is not None:
                return parent
        return None

    def __str__(self):
        _, total, visits = self.value
        return f"SearchNode(id={self.id}, reward={total}, visits={visits})"


class SearchTree:
    """
    Wrapper around MCTS root node with basic bookkeeping.

    Implementation of the Monte Carlo Tree Search (MCTS)
    data structure.
    """

    def __init__(self, root_node):
        self.root = root_node
        self._counter = 1
        self.leaf_states = [root_node.value[0]]

    def insert_child(self, parent_node, child_node):
        """
        Attach child_node under parent_node and maintain leaf_states list.
        """
        child_node.id = self._counter
        self._counter += 1

        parent_node.add_child(child_node)
        if parent_node.value[0] in self.leaf_states:
            self.leaf_states.remove(parent_node.value[0])
        parent_node.is_leaf = False
        self.leaf_states.append(child_node.value[0])

    def parent_of(self, node):
        if node is self.root:
            return None
        return self.root.find_parent_of(node)

    def back_propagate(self, reward, node):
        """
        Add reward and visit count up to the root.

        Backpropagation step in MCTS:
          - Starting from the leaf, we move up to the root,
            updating (Q, N) for each node on the path:

                Q_new = Q_old + reward
                N_new = N_old + 1
        """
        state, total_reward, visits = node.value
        node.value = (state, total_reward + reward, visits + 1)

        parent = self.parent_of(node)
        if parent is not None:
            self.back_propagate(reward, parent)

    def select_leaf(self, node=None):
        """
        Follow UCB from root until a leaf node is found.

        The Selection step in MCTS:
          - Repeatedly select the child with maximal UCB1
            until reaching a node marked as a leaf.
        """
        if node is None:
            node = self.root
        if not node.is_leaf:
            return self.select_leaf(node.select_child_ucb())
        return node


#########################
# Base Reflex Agent     #
#########################


class ReflexCaptureAgent(CaptureAgent):
    """
    Shared logic for offensive and defensive agents:
      - tunnel detection & map partitioning
      - simple MCTS escape routine when “stuck”
      - convenient wrappers to the contest API

    Evaluation is feature-based:

      Q(s, a) = Σ_i w_i * f_i(s, a)

    where:
      - f_i(s, a) are features extracted from (state, action)
      - w_i are the learned or hand-tuned weights
    """

    ############ Initialization ############

    def register_initial_state(self, game_state):
        """
        Called once at game start.
        """
        self.initial_position = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # High-level behavior state
        self.switch_entrance_flag = False
        self.next_entrance_target = None
        self.carried_food_count = 0
        self.last_tunnel_entry = None

        global MAP_WALLS, MAP_TUNNELS, OPEN_ROADS, FREE_CELLS, HOME_TUNNELS
        MAP_WALLS = game_state.get_walls().as_list()

        if not MAP_TUNNELS:
            FREE_CELLS = [p for p in game_state.get_walls().as_list(False)]
            MAP_TUNNELS = compute_all_tunnels(FREE_CELLS)
            OPEN_ROADS = list(set(FREE_CELLS) - set(MAP_TUNNELS))

        # per-agent rolling state
        self.selected_capsule = None
        self.preferred_open_food = None
        self.preferred_tunnel_food = None
        self.boundary_objective = None
        self.stuck_counter = 0
        self.last_lost_food = None
        self.mcts_escape_mode = False

        # compute defensive tunnels on our side only once
        layout_width = game_state.data.layout.width
        red_side_cells = [p for p in FREE_CELLS if p[0] < layout_width / 2]
        blue_side_cells = [p for p in FREE_CELLS if p[0] >= layout_width / 2]
        if not HOME_TUNNELS:
            HOME_TUNNELS[:] = compute_all_tunnels(red_side_cells if self.red else blue_side_cells)

    # older name used by some frameworks
    def registerInitialState(self, game_state):
        return self.register_initial_state(game_state)

    def choose_action(self, game_state):
        """
        Primary decision function used by the engine:
          - default: reflex evaluation over legal actions
          - if marked as “stuck”: trigger MCTS simulation

        Reflex evaluation:
          For each action a:
            1) Extract features f_i(s,a)
            2) Compute Q(s,a) = Σ_i w_i * f_i(s,a)
          Then select the action a* with maximal Q(s,a).
        """
        legal_actions = game_state.get_legal_actions(self.index)
        action_scores = [self.evaluate(game_state, act) for act in legal_actions]
        best_value = max(action_scores)

        if self.mcts_escape_mode:
            return self.simulation(game_state)

        best_actions = [act for act, score in zip(legal_actions, action_scores)
                        if score == best_value]
        return random.choice(best_actions)

    # alias for compatibility with baseline API
    def chooseAction(self, game_state):
        return self.choose_action(game_state)

    def get_successor(self, game_state, action):
        """
        Generate successor state and (if necessary) step until we land on grid.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            successor = successor.generate_successor(self.index, action)
        return successor

    # legacy alias
    def getSuccessor(self, game_state, action):
        return self.get_successor(game_state, action)

    def evaluate(self, game_state, action):
        """
        Linear evaluation function:

            Q(s, a) = Σ_i w_i * f_i(s, a)

        implemented via:
          - self.getFeatures(s, a) -> feature vector (util.Counter)
          - self.getWeights(s, a)  -> weight vector
        """
        features = self.getFeatures(game_state, action)
        weights = self.getWeights(game_state, action)
        return features * weights

    #  Framework API wrappers from CaptureAgent  #

    def getFood(self, game_state):
        return self.get_food(game_state)

    def getCapsules(self, game_state):
        return self.get_capsules(game_state)

    def getOpponents(self, game_state):
        return self.get_opponents(game_state)

    def getFoodYouAreDefending(self, game_state):
        return self.get_food_you_are_defending(game_state)

    def getCapsulesYouAreDefending(self, game_state):
        return self.get_capsules_you_are_defending(game_state)

    def getPreviousObservation(self):
        return self.get_previous_observation()

    def getCurrentObservation(self):
        return self.get_current_observation()

    def getMazeDistance(self, pos1, pos2):
        return self.get_maze_distance(pos1, pos2)

    ############ Tunnel & boundary helpers ############

    def remaining_time(self, game_state):
        return game_state.data.timeleft

    def _boundary_cells(self, game_state):
        """
        Return lists of boundary cells for red and blue teams.

        The boundary is the vertical split line:
          - red_boundary = cells on the red side (x = width/2 - 1)
          - blue_boundary = cells on the blue side (x = width/2)
        """
        width = game_state.data.layout.width
        all_legal = [p for p in game_state.get_walls().as_list(False)]
        red_boundary = [p for p in all_legal if p[0] == width / 2 - 1]
        blue_boundary = [p for p in all_legal if p[0] == width / 2]
        return red_boundary, blue_boundary

    def tunnel_food_distance(self, game_state, successor_state):
        """
        If we step from non-tunnel into a tunnel, return distance to
        the next food cell inside that tunnel component; otherwise 0.

        Internally, this uses a DFS-style search with an explicit stack:
          - Each stack element carries (position, depth).
          - The first time we encounter a food cell, we return its depth.
        """
        current_pos = game_state.get_agent_state(self.index).get_position()
        succ_pos = successor_state.get_agent_state(self.index).get_position()

        if current_pos not in MAP_TUNNELS and succ_pos in MAP_TUNNELS:
            self.last_tunnel_entry = current_pos
            stack = util.Stack()
            visited = set()
            stack.push((succ_pos, 1))

            while not stack.is_empty():
                (x, y), path_len = stack.pop()
                if self.getFood(game_state)[int(x)][int(y)]:
                    return path_len

                if (x, y) in visited:
                    continue
                visited.add((x, y))

                for nxt in list_neighbors((x, y), MAP_TUNNELS):
                    if nxt not in visited:
                        stack.push((nxt, path_len + 1))
        return 0

    def find_tunnel_food(self, game_state):
        """
        BFS inside tunnel from our current position to locate nearest food.

        Breadth-First Search (BFS) algorithm:
          - Initialize a FIFO queue with the starting position.
          - Repeatedly pop from the queue, pushing all unvisited neighbors.
          - The first time we see a food cell, it is guaranteed to be
            at minimum distance in an unweighted grid.
        """
        start_pos = game_state.get_agent_state(self.index).get_position()
        queue = util.Queue()
        visited = set()
        queue.push(start_pos)

        while not queue.is_empty():
            x, y = queue.pop()
            if self.getFood(game_state)[int(x)][int(y)]:
                return (x, y)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            for nxt in list_neighbors((x, y), MAP_TUNNELS):
                if nxt not in visited:
                    queue.push(nxt)
        return None

    def getEntrance(self, game_state):
        """
        Compute legal entrance cells along the middle boundary
        (one set for red, the other for blue).

        We pair cells on x = width/2 - 1 with cells on x = width/2
        that share the same y-coordinate and are both legal.
        """
        width = game_state.data.layout.width
        all_legal = [p for p in game_state.get_walls().as_list(False)]

        left_column = [p for p in all_legal if p[0] == width / 2 - 1]
        right_column = [p for p in all_legal if p[0] == width / 2]

        red_entrances = []
        blue_entrances = []
        for left in left_column:
            for right in right_column:
                if left[0] + 1 == right[0] and left[1] == right[1]:
                    red_entrances.append(left)
                    blue_entrances.append(right)

        return red_entrances if self.red else blue_entrances

    ############ MCTS logic ############

    def run_rollout(self, game_state):
        """
        Rollout for MCTS:
          - perform up to 20 random actions
          - heavy negative reward if we collide with a ghost

        The Simulation (Rollout) phase in MCTS:
          - We simulate a random policy π_rand for a fixed horizon H = 20.
          - Reward is estimated from the terminal or last state.
        """
        remaining_steps = 20
        enemy_states = [game_state.get_agent_state(i) for i in self.getOpponents(game_state)]
        ghost_states = [e for e in enemy_states if not e.is_pacman and e.get_position() is not None]
        ghost_positions = [g.get_position() for g in ghost_states]

        current_state = game_state
        while remaining_steps > 0:
            remaining_steps -= 1
            legal_actions = current_state.get_legal_actions(self.index)
            chosen = random.choice(legal_actions)
            successor = self.get_successor(current_state, chosen)

            my_pos = move_in_direction(current_state.get_agent_state(self.index).get_position(), chosen)
            if my_pos in ghost_positions:
                return -9999

            current_state = successor

        return self.evaluate(current_state, Directions.STOP)

    def simulation(self, game_state):
        """
        Run MCTS for ~0.95 seconds and return a direction
        corresponding to the best child of the root.

        Complete MCTS loop:
          1) Selection      - using UCB1 on the tree (select_leaf)
          2) Expansion      - expand_node
          3) Simulation     - run_rollout
          4) Backpropation  - back_propagate
        """
        start_x, start_y = game_state.get_agent_position(self.index)
        root_node = SearchNode((game_state, 0, 0))
        tree = SearchTree(root_node)

        start_time = time.time()
        while time.time() - start_time < 0.95:
            self.iteration(tree)

        best_child_state = tree.root.select_child_ucb().value[0]
        next_x, next_y = best_child_state.get_agent_position(self.index)

        if next_x == start_x + 1:
            return Directions.EAST
        if next_x == start_x - 1:
            return Directions.WEST
        if next_y == start_y + 1:
            return Directions.NORTH
        if next_y == start_y - 1:
            return Directions.SOUTH
        return Directions.STOP

    def iteration(self, search_tree):
        """
        A single MCTS iteration:
          - select a leaf
          - expand_node if needed
          - rollout
          - back-propagate the reward

          function MCTS(root):
              while time not over:
                  leaf = select(root)
                  if leaf not fully expanded:
                      child = expand(leaf)
                      reward = simulate(child)
                  else:
                      reward = simulate(leaf)
                  backpropagate(reward, leaf)
        """
        root = search_tree.root
        if not root.children:
            self.expand_node(search_tree, root)
            return

        leaf = search_tree.select_leaf()
        _, _, visit_count = leaf.value

        # first visit: just rollout and back up
        if visit_count == 0:
            reward = self.run_rollout(leaf.value[0])
            search_tree.back_propagate(reward, leaf)
        # second visit: expand children, then rollout from one of them
        elif visit_count == 1:
            self.expand_node(search_tree, leaf)
            chosen_child = random.choice(leaf.children)
            reward = self.run_rollout(chosen_child.value[0])
            search_tree.back_propagate(reward, chosen_child)

    def expand_node(self, search_tree, node):
        """
        Expand a node by generating children for all non-STOP actions.

        The Expansion phase in MCTS:
          - For each legal action a (except STOP), we create a child node
            with successor state s' = successor(s, a).
        """
        state = node.value[0]
        actions = state.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        for action in actions:
            succ_state = state.generate_successor(self.index, action)
            child_node = SearchNode((succ_state, 0, 0))
            search_tree.insert_child(node, child_node)


################################
# Offensive Reflex Agent       #
################################


class OffensiveReflexAgent(ReflexCaptureAgent):

    def getFeatures(self, game_state, action):
        """
        Offensive features:
          - score, food distances, ghost avoidance
          - tunnel / capsule handling
          - basic “stuck” behavior with boundary entrances

        Features f_i(s, a) are used in the evaluation function:

            Q(s, a) = Σ_i w_i * f_i(s, a)
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        successor_pos = successor.get_agent_state(self.index).get_position()
        stepped_pos = move_in_direction(current_pos, action)

        # opponent info
        enemies = [game_state.get_agent_state(i) for i in self.getOpponents(game_state)]
        close_ghosts = [
            e for e in enemies
            if (not e.is_pacman) and e.get_position() is not None
            and manhattan_distance(current_pos, e.get_position()) <= 5
        ]
        scared_ghosts = [g for g in close_ghosts if g.scared_timer > 1]
        active_ghosts = [g for g in close_ghosts if g not in scared_ghosts]

        # food and capsules
        food_cells = self.getFood(game_state).as_list()
        open_food = [f for f in food_cells if f not in MAP_TUNNELS]
        tunnel_food = [f for f in food_cells if f in MAP_TUNNELS]

        capsules = self.getCapsules(game_state)
        tunnel_distance = self.tunnel_food_distance(game_state, successor)

        # basic score feature
        features['successorScore'] = self.get_score(successor)

        # clear targets when ghosts disappear
        if not close_ghosts:
            self.selected_capsule = None
            self.preferred_open_food = None
            self.preferred_tunnel_food = None

        # reset entrance flag when we are pacman
        if current_state.is_pacman:
            self.switch_entrance_flag = False

        # track amount of food we carry
        if stepped_pos in food_cells:
            self.carried_food_count += 1
        if not current_state.is_pacman:
            self.carried_food_count = 0

        # force going home when time is low
        if self.remaining_time(game_state) / 4 < self.getLengthToHome(game_state) + 3:
            features['home_distance'] = self.getLengthToHome(successor)
            return features

        # greedily go to food when there are no active ghosts
        if not active_ghosts and food_cells and len(food_cells) >= 3:
            features['nearest_food'] = min(self.getMazeDistance(successor_pos, f) for f in food_cells)
            if successor_pos in food_cells:
                features['nearest_food'] = -1

        # when food is almost finished, head home
        if len(food_cells) < 3:
            features['return'] = self.getLengthToHome(successor)

        # ghosts nearby and plenty of food left
        if active_ghosts and len(food_cells) >= 3:
            min_ghost = min(self.getMazeDistance(successor_pos, g.get_position()) for g in active_ghosts)
            features['ghost_threat'] = 100 - min_ghost

            ghost_positions = [g.get_position() for g in active_ghosts]
            if stepped_pos in ghost_positions:
                features['will_die'] = 1
            if ghost_positions:
                neighbor_cells = list_neighbors(ghost_positions[0], FREE_CELLS)
                if stepped_pos in neighbor_cells:
                    features['will_die'] = 1

            if open_food:
                features['open_food_dist'] = min(self.getMazeDistance(successor_pos, f) for f in open_food)
                if successor_pos in open_food:
                    features['open_food_dist'] = -1
            else:
                features['return'] = self.getLengthToHome(successor)

        # choose “safe” open-road food (closer to us than to any ghost)
        if active_ghosts and len(food_cells) >= 3 and open_food:
            safe_targets = []
            for food in open_food:
                my_dist = self.getMazeDistance(current_pos, food)
                ghost_dists = [self.getMazeDistance(g.get_position(), food) for g in active_ghosts]
                if my_dist < min(ghost_dists):
                    safe_targets.append(food)

            if safe_targets:
                best_dist = min(self.getMazeDistance(current_pos, f) for f in safe_targets)
                for f in safe_targets:
                    if self.getMazeDistance(current_pos, f) == best_dist:
                        self.preferred_open_food = f
                        break

        # choose safe tunnel food if possible
        if active_ghosts and tunnel_food and not scared_ghosts and len(food_cells) >= 3:
            safe_tunnel_targets = []
            for tf in tunnel_food:
                entry = tunnel_entry_for(tf, MAP_TUNNELS, FREE_CELLS)
                my_len = self.getMazeDistance(current_pos, tf) + self.getMazeDistance(tf, entry)
                ghost_len = min(self.getMazeDistance(g.get_position(), entry) for g in active_ghosts)
                if my_len < ghost_len:
                    safe_tunnel_targets.append(tf)

            if safe_tunnel_targets:
                best_dist = min(self.getMazeDistance(current_pos, f) for f in safe_tunnel_targets)
                for f in safe_tunnel_targets:
                    if self.getMazeDistance(current_pos, f) == best_dist:
                        self.preferred_tunnel_food = f
                        break

        # go towards selected “safe” open-road food
        if self.preferred_open_food is not None:
            features['safe_food_dist'] = self.getMazeDistance(successor_pos, self.preferred_open_food)
            if successor_pos == self.preferred_open_food:
                features['safe_food_dist'] = -0.0001
                self.preferred_open_food = None

        # if no open-road target, try chosen tunnel food instead
        if features['safe_food_dist'] == 0 and self.preferred_tunnel_food is not None:
            features['safe_food_dist'] = self.getMazeDistance(successor_pos, self.preferred_tunnel_food)
            if successor_pos == self.preferred_tunnel_food:
                features['safe_food_dist'] = 0
                self.preferred_tunnel_food = None

        # capsule logic: go for capsule if we win the race against ghosts
        if active_ghosts and capsules:
            for c in capsules:
                my_dist = self.getMazeDistance(current_pos, c)
                ghost_dists = [self.getMazeDistance(c, g.get_position()) for g in active_ghosts]
                if my_dist < min(ghost_dists):
                    self.selected_capsule = c

        if scared_ghosts and capsules:
            for c in capsules:
                my_dist = self.getMazeDistance(current_pos, c)
                ghost_dists = [self.getMazeDistance(c, g.get_position()) for g in scared_ghosts]
                if my_dist >= scared_ghosts[0].scared_timer and my_dist < min(ghost_dists):
                    self.selected_capsule = c

        if current_pos in MAP_TUNNELS:
            for c in capsules:
                if c in tunnel_component(current_pos, MAP_TUNNELS):
                    self.selected_capsule = c

        if self.selected_capsule is not None:
            features['capsule_dist'] = self.getMazeDistance(successor_pos, self.selected_capsule)
            if successor_pos == self.selected_capsule:
                features['capsule_dist'] = 0
                self.selected_capsule = None

        # avoid triggering capsule when there are no active ghosts
        if not active_ghosts and successor_pos in capsules:
            features['avoid_capsule'] = 0.1

        if action == Directions.STOP:
            features['stop'] = 1

        # avoid entering “empty” tunnels (no food ahead)
        if (successor.get_agent_state(self.index).is_pacman and
                current_pos not in MAP_TUNNELS and
                successor.get_agent_state(self.index).get_position() in MAP_TUNNELS and
                tunnel_distance == 0):
            features['empty_tunnel'] = -1

        # avoid long tunnel if ghosts are too near
        if active_ghosts:
            closest_ghost = min(self.getMazeDistance(current_pos, g.get_position()) for g in active_ghosts)
            if tunnel_distance != 0 and tunnel_distance * 2 >= closest_ghost - 1:
                features['waste_tunnel'] = -1

        if scared_ghosts:
            closest_scared = min(self.getMazeDistance(current_pos, g.get_position()) for g in scared_ghosts)
            if tunnel_distance != 0 and tunnel_distance * 2 >= scared_ghosts[0].scared_timer - 1:
                features['waste_tunnel'] = -1

        # escape tunnels that become dangerous
        if current_pos in MAP_TUNNELS and active_ghosts:
            food_pos = self.find_tunnel_food(game_state)
            if food_pos is None:
                features['escape_tunnel'] = self.getMazeDistance(
                    move_in_direction(current_pos, action),
                    self.last_tunnel_entry
                )
            else:
                path_len = (self.getMazeDistance(successor_pos, food_pos) +
                            self.getMazeDistance(food_pos, self.last_tunnel_entry))
                ghost_len = min(self.getMazeDistance(self.last_tunnel_entry, g.get_position())
                                for g in active_ghosts)
                if ghost_len - path_len <= 1 and not scared_ghosts:
                    features['escape_tunnel'] = self.getMazeDistance(
                        move_in_direction(current_pos, action),
                        self.last_tunnel_entry
                    )

        if current_pos in MAP_TUNNELS and scared_ghosts:
            food_pos = self.find_tunnel_food(game_state)
            if food_pos is None:
                features['escape_tunnel'] = self.getMazeDistance(
                    move_in_direction(current_pos, action),
                    self.last_tunnel_entry
                )
            else:
                path_len = (self.getMazeDistance(successor_pos, food_pos) +
                            self.getMazeDistance(food_pos, self.last_tunnel_entry))
                if scared_ghosts[0].scared_timer - path_len <= 1:
                    features['escape_tunnel'] = self.getMazeDistance(
                        move_in_direction(current_pos, action),
                        self.last_tunnel_entry
                    )

        # simple “stuck” heuristic on defense
        if (not current_state.is_pacman and active_ghosts and self.stuck_counter != -1):
            self.stuck_counter += 1

        if current_state.is_pacman or successor_pos == self.next_entrance_target:
            self.stuck_counter = 0
            self.next_entrance_target = None

        if self.stuck_counter > 10:
            self.stuck_counter = -1
            self.next_entrance_target = random.choice(self.getEntrance(game_state))

        if self.next_entrance_target is not None and features['safe_food_dist'] == 0:
            features['boundary_run'] = self.getMazeDistance(successor_pos, self.next_entrance_target)

        return features

    def getWeights(self, game_state, action):
        return {
            'successorScore': 1,
            'home_distance': -100,
            'nearest_food': -2,
            'open_food_dist': -3,
            'ghost_threat': -10,
            'will_die': -1000,
            'safe_food_dist': -11,
            'capsule_dist': -1200,
            'return': -1,
            'avoid_capsule': -1,
            'stop': -50,
            'empty_tunnel': 100,
            'waste_tunnel': 100,
            'escape_tunnel': -1001,
            'boundary_run': -1001,
        }

    def getLengthToHome(self, game_state):
        """
        Return distance from current position to the middle boundary
        on our own side.

        Implemented using the maze distance (shortest path in the grid).
        """
        pos = game_state.get_agent_state(self.index).get_position()
        red_boundary, blue_boundary = self._boundary_cells(game_state)
        targets = red_boundary if self.red else blue_boundary
        return min(self.getMazeDistance(pos, b) for b in targets)


################################
# Defensive Reflex Agent       #
################################


class DefensiveReflexAgent(ReflexCaptureAgent):

    def getLengthToBoundary(self, game_state):
        """
        Distance from current position to the boundary on our side.
        """
        pos = game_state.get_agent_state(self.index).get_position()
        red_boundary, blue_boundary = self._boundary_cells(game_state)
        targets = red_boundary if self.red else blue_boundary
        return min(self.getMazeDistance(pos, b) for b in targets)

    def getFeatures(self, game_state, action):
        """
        Defensive feature extraction:

          - Encourage staying on defense (on_defense)
          - Chase visible invaders (invader_distance)
          - Protect capsules (capsule_protection)
          - React to lost food (lost_food_distance)
          - Avoid wasteful tunnel moves (waste_tunnel)

        Combined again via Q(s, a) = Σ_i w_i * f_i(s, a).
        """
        features = util.Counter()

        successor = self.get_successor(game_state, action)
        current_pos = game_state.get_agent_state(self.index).get_position()
        current_state = game_state.get_agent_state(self.index)
        succ_state = successor.get_agent_state(self.index)
        succ_pos = succ_state.get_position()

        defended_capsules = self.getCapsulesYouAreDefending(game_state)

        # positive bias when we are on defense (not pacman)
        features['on_defense'] = 0 if succ_state.is_pacman else 100

        if self.boundary_objective is None:
            features['run_to_boundary'] = self.getLengthToBoundary(successor)

        if self.getLengthToBoundary(successor) <= 2:
            self.boundary_objective = 0

        # enemy info
        enemies_succ = [successor.get_agent_state(i) for i in self.getOpponents(successor)]
        enemies_curr = [game_state.get_agent_state(i) for i in self.getOpponents(game_state)]

        invaders_succ = [e for e in enemies_succ if e.is_pacman and e.get_position() is not None]
        invaders_curr = [e for e in enemies_curr if e.is_pacman and e.get_position() is not None]

        # block tunnels if exactly one invader is inside and we can beat it to the entry
        if self.should_block_tunnel(invaders_curr, current_pos, defended_capsules) and current_state.scared_timer == 0:
            entry_cell = tunnel_entry_for(invaders_curr[0].get_position(), MAP_TUNNELS, FREE_CELLS)
            if entry_cell is not None:
                features['tunnel_block'] = self.getMazeDistance(entry_cell, succ_pos)
                return features

        # leave defensive tunnels when there are no invaders
        if current_pos in HOME_TUNNELS and not invaders_curr:
            features['leaveTunnel'] = self.getMazeDistance(self.initial_position, succ_pos)

        features['invader_count'] = len(invaders_succ)

        # avoid entering tunnels if there is no threat and we are not scared
        if not invaders_curr and not succ_state.is_pacman and current_state.scared_timer == 0:
            if current_pos not in HOME_TUNNELS and succ_pos in HOME_TUNNELS:
                features['waste_tunnel'] = -1

        # chase invaders when not scared
        if invaders_succ and current_state.scared_timer == 0:
            dists = [self.getMazeDistance(succ_pos, inv.get_position()) for inv in invaders_succ]
            features['invader_distance'] = min(dists)
            features['boundary_distance'] = self.getLengthToBoundary(successor)

        # in scared mode, keep some distance but follow
        if invaders_succ and current_state.scared_timer != 0:
            dists = [self.getMazeDistance(succ_pos, inv.get_position()) for inv in invaders_succ]
            min_d = min(dists)
            # follow_spacing penalizes being too close or too far from ideal distance ≈ 2
            features['follow_spacing'] = (min_d - 2) ** 2
            if current_pos not in HOME_TUNNELS and succ_pos in HOME_TUNNELS:
                features['waste_tunnel'] = -1

        if invaders_succ and defended_capsules:
            d_caps = [self.getMazeDistance(c, succ_pos) for c in defended_capsules]
            features['capsule_protection'] = min(d_caps)

        if action == Directions.STOP:
            features['stop'] = 1

        reverse_dir = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_dir:
            features['reverse'] = 1

        # react to lost defended food
        if self.getPreviousObservation() is not None:
            lost_food_cell = self.find_lost_food()
            if not invaders_succ and lost_food_cell is not None:
                self.last_lost_food = lost_food_cell

            if self.last_lost_food is not None and not invaders_succ:
                features['lost_food_distance'] = self.getMazeDistance(succ_pos, self.last_lost_food)

            if succ_pos == self.last_lost_food or invaders_succ:
                self.last_lost_food = None

        return features

    def getWeights(self, game_state, action):
        return {
            'invader_count': -100,
            'on_defense': 10,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
            'boundary_distance': -3,
            'capsule_protection': -3,
            'waste_tunnel': 200,
            'follow_spacing': -100,
            'tunnel_block': -10,
            'leaveTunnel': -0.1,
            'run_to_boundary': -2,
            'lost_food_distance': -1,
        }

    def should_block_tunnel(self, current_invaders, my_position, defended_capsules):
        """
        Return True if exactly one invader is inside a tunnel and
        we can reach the tunnel entry no later than them, and the
        defended capsule is not in that tunnel.

        Logic:
          - If there is a single invader in MAP_TUNNELS, compute:
                entry = tunnel_entry_for(invader_pos)
                d_me   = dist(entry, my_position)
                d_inv  = dist(entry, invader_pos)
            If d_me <= d_inv and the capsule is not inside that tunnel
            component, then we commit to blocking that entry.
        """
        if len(current_invaders) == 1:
            inv_pos = current_invaders[0].get_position()
            if inv_pos in MAP_TUNNELS:
                component = tunnel_component(inv_pos, MAP_TUNNELS)
                entry_cell = tunnel_entry_for(inv_pos, MAP_TUNNELS, FREE_CELLS)
                if entry_cell is None:
                    return False
                my_d = self.getMazeDistance(entry_cell, my_position)
                inv_d = self.getMazeDistance(entry_cell, inv_pos)

                # true only if NO defended capsule lies in that tunnel component
                capsule_in_tunnel = any(c in component for c in defended_capsules)

                if my_d <= inv_d and not capsule_in_tunnel:
                    return True
        return False


    def find_lost_food(self):
        """
        Return location of lost defended food between last and current state, if any.
        """
        prev_state = self.getPreviousObservation()
        curr_state = self.getCurrentObservation()

        if prev_state is None or curr_state is None:
            return None

        prev_food = self.getFoodYouAreDefending(prev_state).as_list()
        curr_food = self.getFoodYouAreDefending(curr_state).as_list()

        if len(curr_food) < len(prev_food):
            for loc in prev_food:
                if loc not in curr_food:
                    return loc
        return None
