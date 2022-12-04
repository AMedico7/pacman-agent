# myTeam.py
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


import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from game import Directions, Actions

import pickle

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AttackAgent', second='DefenceAgent', num_training=0):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

# Superclass from which the attacker agent inherits from

# Our idea is the following: If we are very far ahead the attack agent will become a defence agent, so we can keep the other team from winning (hopefully, we'll keep them from scoring)

class DefenceAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.init_state = None


        # Even indexes correspond to the read team, while odd indexes correspond to blue team
        self.red = True if self.index%2 == 0 else False

        self.higher_half = True if self.index%4 < 2 else False

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        self.init_state = game_state

        # To know the boundary, this will be used to know when we have to stop retreating, as well as for the defence agent to "camp" the boundary
        if self.red:
            boundary = (game_state.data.layout.width - 2) / 2
        else:
            boundary = ((game_state.data.layout.width - 2) / 2) + 1
        boundary = int(boundary)

        self.boundary = [(boundary, i) for i in range(1, game_state.data.layout.height - 1) if
                         not game_state.has_wall(boundary, i)]

        self.my_boundary = []
        if self.higher_half:
            self.my_boundary = [(boundary, i) for i in range(int(game_state.data.layout.height/2), game_state.data.layout.height - 1) if not game_state.has_wall(boundary, i)]
        else:
            self.my_boundary = [(boundary, i) for i in range(1, int(game_state.data.layout.height / 2)) if not game_state.has_wall(boundary, i)]

        self.winning = False

        # Patrol point around which the agent will walk
        self.target = random.choice(self.boundary)
        self.past_food = self.get_food_you_are_defending(game_state).as_list()
        self.targeting_inv = False


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def final(self, game_state):
        super().final(game_state)

    def are_we_winning(self, game_state):
        if self.red:
            self.winning = True if self.get_score(game_state) >= 5 else False
        else:
            self.winning = True if self.get_score(game_state) <= -5 else False

    def get_best_action(self,game_state):
        actions = game_state.get_legal_actions(self.index)
        agent_state = game_state.get_agent_state(self.index)

        best_actions = []
        best_dist = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)

            # We don't want for the agent to stop, nether for it to go to the enemy side
            if action == 'Stop' or (self.red and (pos2[0] > self.boundary[0][0])) or (
                    not self.red and pos2[0] < self.boundary[0][0]):
                continue
            dist = self.get_maze_distance(self.target, pos2)

            if dist == best_dist:
                best_actions.append(action)

            elif best_dist > dist:
                best_actions = [action]
                best_dist = dist

        action = random.choice(best_actions)
        return action

    def choose_action(self, game_state):

        self.are_we_winning(game_state)
        #print(actions)
        agent_state = game_state.get_agent_state(self.index)

        cur_pos = agent_state.get_position()
        # Positions of the dots that we are defending
        dots_defending = self.get_food_you_are_defending(game_state).as_list()

        pos2 = game_state.get_agent_position(self.index)

        # There is a visible invader -> chase it
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        vis_invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # We only chase it if we are not scared, otherwise we will suicide
        if len(vis_invaders)>0 and agent_state.scared_timer == 0:
            self.targeting_inv = True
            positions = [invader.get_position() for invader in vis_invaders]
            self.target = min(positions, key=lambda x: self.get_maze_distance(pos2, x))

        # Invaders are not visible, but a dot was eaten -> go there
        elif len(self.past_food) > len(dots_defending):
            self.target = (list(set(self.past_food)- set(dots_defending)).pop())

        # If we have very few dots (5 or less), we will patrol around them
        elif len(dots_defending) <= 5:
            points = dots_defending + self.get_capsules_you_are_defending(game_state)
            self.target = random.choice(points)

        action = self.get_best_action(game_state)

        cur_pos = game_state.get_agent_position(self.index)

        # Walk towards different points of the boundary each time we reach the target point of the target or we kill a pacman:
        if cur_pos == self.target:
            self.targeting_inv = False if self.targeting_inv else True
            self.target = random.choice(self.my_boundary) if self.winning else random.choice(self.boundary)

        # Update the food attribute
        self.past_food = dots_defending

        return action

        # return super().choose_action(game_state)



class AttackAgent(DefenceAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Attribute to know the amount of food that we need to eat from the board at any point of the game
        self.goal_food = len(self.get_food(self.init_state).as_list())
        # Used to know whether we are retreating or not
        self.retreating = False

        # RELATED TO APPROX Q-LEARNING
        # When training the agent, we used epsilon 0.4, however, after having trained it with enough matches for the agent to be decent, we want to exploit -> 0
        self.epsilon = 0.0
        self.alpha = 0.1
        self.discount = 0.9
        self.numTraining = 1000
        self.numIters = 0

        self.weights = {'bias': -1.7481743905295866, 'carrying': 7.201896785569032, 'dist_to_food': -7.423967243663026, 'dist_to_capsule': -4.425434892857854, 'ghost_distance': 8.32367450007373, 'invader_distance': 3.632439875193794e-06}
        # with open('attack_weights.pickle', 'rb') as handle:
        #     self.weights = pickle.load(handle)

        self.dead_ends = []
        self.starting = True

    def final(self, game_state):
        super().final(game_state)
        # print(self.weights)
        # with open('attack_weights.pickle', 'wb') as handle:
        #     pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_qValue(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)

        return features * weights

    def update(self, game_state, action, next_state, reward):
        features = self.get_features(game_state, action)
        oldValue = self.get_qValue(game_state, action)
        newValue = self.getValue(next_state)

        reward = self.getReward(game_state, action)

        difference = (reward + self.discount * newValue) - oldValue

        for feature in features:
            newWeight = self.alpha * difference * features[feature]

            self.weights[feature] += newWeight
        # print(self.weights)

    def updateWeights(self, game_state, action):
        w_invader_distance = self.weights['invader_distance']

        nextState = self.get_successor(game_state, action)

        reward = self.getReward(game_state, action)
        self.update(game_state, action, nextState, reward)

        # If we are not in our territory, we do not really want to take our current state into account when updating the weight for the invader distance
        if game_state.get_agent_state(self.index).is_pacman:
            self.weights['invader_distance'] = w_invader_distance

    def computeValueFromQValues(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        temp_max = -999999

        if len(legal_actions) == 0:
            return 0.0
        bestAction = self.getPolicy(game_state)
        return self.get_qValue(game_state, bestAction)

    # Q-learning related
    def computeActionFromQValues(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)

        if len(legal_actions) == 0:
            return 'Stop'

        best_Q = float('-inf')
        bestActions = []

        for action in legal_actions:
            cur_Q  = self.get_qValue(game_state, action)

            if cur_Q > best_Q:
                best_Q = cur_Q

                bestActions = [action]
            elif cur_Q == best_Q:
                bestActions.append(action)
        action = random.choice(bestActions)
        # print(action)
        return action


    def getPolicy(self, game_state):
        return self.computeActionFromQValues(game_state)

    def getValue(self, game_state):
        return self.computeValueFromQValues(game_state)

    def choose_action(self, game_state):

        # The agent will retreat to its own territory if he is carrying lots of dots
        # This way we can guarantee some points even if we do not eat them all at once
        self.are_we_winning(game_state)

        if self.winning:
            return super().choose_action(game_state)

        agent_state = game_state.get_agent_state(self.index)
        cur_pos = agent_state.get_position()

        if cur_pos == self.start:
            self.starting = True
            self.target = random.choice(self.boundary)

        if cur_pos == self.target:
            self.starting = False

        if self.starting:
            return super().get_best_action(game_state)

        actions = game_state.get_legal_actions(self.index)

        if self.retreating:
            carrying = agent_state.num_carrying

            # If the agent is retreating but it is not carrying food, it can be that it has deposited them or that it was eaten
            if carrying == 0:
                self.retreating = False
                # If the agent is in the boundary, then it means that it has safely deposited the dots it was carrying to his side, otherwise it means it was eaten on its way there
                if agent_state.get_position()[0] == self.boundary:
                    # We update the goal food
                    g_food = (self.total_food - carrying)
                    self.goal_food = g_food if (g_food/2) >= 5 else self.goal_food

            # We want to go back to our territory and deposit the food we've eaten so far
            else:
                best_dist = float('inf')
                best_actions = []

                game_area = (game_state.data.layout.width * game_state.data.layout.height)

                for action in actions:
                    successor = self.get_successor(game_state, action)

                    pos2 = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(self.start, pos2) / game_area

                    # We want to make the distance to the start smaller, but we don't want to be eaten (if we are eaten the we will be at the start, so distance = 0)
                    if dist != 0:
                        if dist == best_dist:
                            best_actions.append(action)

                        elif best_dist > dist:
                            best_actions = [action]
                            best_dist = dist
                return random.choice(best_actions)

        # print("not retreating")
        if len(actions) == 0:
            return None

        if self.numIters < self.numTraining:
            self.numIters += 1
            for action in actions:
                self.updateWeights(game_state, action)

        action = None
        if util.flipCoin(self.epsilon):
            # Explore
            action = random.choice(actions)
        else:
            # Exploit
            action = self.getPolicy(game_state)
        return action


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        features = util.Counter()
        features['bias'] = 1.0

        successor = self.get_successor(game_state, action)
        # features['successor_score'] = self.get_score(successor)

        agent_state = successor.get_agent_state(self.index)
        cur_pos = agent_state.get_position()

        # We will divide the distances by the game area so that the weights do not diverge when updating them
        game_area = (game_state.data.layout.width * game_state.data.layout.height)

        # Compute distance to the closest ghost (visible)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        visible = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        not_scared = False in [enemy.scared_timer >= 8 for enemy in visible]


        if len(visible) > 0 and not_scared:
            g_dist = min([self.get_maze_distance(cur_pos, a.get_position()) for a in visible])
        # If there are no visible ghosts, we will get the minimum approximated distance only if we are pacman
        # Otherwise we do not really need to be wary of them unless they are visible (means they are close to us and could eat us just as we get to enemy territory)
        elif agent_state.is_pacman:
            approx_distances = [game_state.get_agent_distances()[i] for i in self.get_opponents(successor)]
            g_dist = min(approx_distances)
        # If they are not visible and we are not pacman, we will make the feature go to 0
        else:
            g_dist = 0
        features['ghost_distance'] = g_dist / game_area

        # Compute distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_food_dist = min([self.get_maze_distance(cur_pos, food) for food in food_list])
            features['dist_to_food'] = min_food_dist / game_area

        # Compute the number of dots that the pacman is carrying
        carrying = agent_state.num_carrying
        features['carrying'] = agent_state.num_carrying

        # Here we check whether we want to retreat or not: we will retreat when the pacman carries a lot of food
        # and the distance to get more food is bigger than the one to go to its territory
        food_left = len(self.get_food(game_state).as_list())

        min_bound_dist = min([self.get_maze_distance(cur_pos, bound) for bound in self.boundary])

        score = self.get_score(game_state)

        # print("carrying: ", carrying >= 1, "visible: ", len(visible) > 0)
        # Retreat if we are in conditions of winning (food_left <=2), if we carry food and there is a ghost nearby (it can reach us in 3 steps) or
        # we have gathered half of the total food and the distance to go back is smaller than going for another piece of food
        if food_left <= 2 or (carrying > 0 and len(visible) > 0 and g_dist <= 3 and not_scared) or (carrying >= self.goal_food/2 and features['dist_to_food'] > min_bound_dist) or (carrying + score >= 5):
            self.retreating = True

        # If we do want to retreat, then we only care about the distance to the ghosts (so that we can get to our side safely)
        # We put all features to 0 so that they are not taken into account when getting the value for that state
        if self.retreating:
            features['bias'] = 0
            features['dist_to_food'] = 0
            features['dist_to_capsule'] = 0
            features['invader_distance'] = 0
            features['carrying'] = 0
            return features

        # Otherwise, we need to compute the value for the rest of the features
        # Compute distance to nearest capsule
        capsule_list = self.get_capsules(successor)
        if len(capsule_list) > 0:
            min_capsule_dist = min([self.get_maze_distance(cur_pos, capsule) for capsule in food_list])
            features['dist_to_capsule'] = min_capsule_dist / game_area

        # Compute distance to closest enemy pacman, this will be used when the agent is in friendly territory and the enemy is visible
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # We are only interested in taking the invader distance into account when there are invaders and when the agent is not a pacman and is not scared
        # Moreover, we are only interested in chasing them if they are very close (3 steps distance), otherwise we will let them go
        if len(invaders) == 0 or agent_state.is_pacman:
            features['invader_distance'] = 0
        elif agent_state.scared_timer == 0:
            dists = [self.get_maze_distance(cur_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)/game_area if min(dists) <= 3 else 0

        # Make the feature values even smaller to control weight divergence
        features.divideAll(10.0)

        return features

    def get_weights(self, game_state, action):
        return self.weights

    def getReward(self, game_state, action):
        successor = self.get_successor(game_state, action)

        cur_agent_state = game_state.get_agent_state(self.index)
        cur_pos = cur_agent_state.get_position()

        next_agent_state = successor.get_agent_state(self.index)
        next_pos = next_agent_state.get_position()

        cur_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]
        new_enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        reward = 0

        # Pacman has deposited the dots it was carrying to its side
        score_diff = self.get_score(successor) - self.get_score(game_state)
        if self.red and score_diff > 0:
            reward = score_diff * 10
        elif not self.red and score_diff < 0:
            reward = -score_diff * 10

        # Pacman has eaten one dot
        carrying = cur_agent_state.num_carrying
        carrying_diff = next_agent_state.num_carrying - carrying
        if carrying_diff > 0:
            reward = 10

        # Pacman has gotten a capsule
        new_scared = [successor.get_agent_state(i) for i in self.get_opponents(successor)][0].scared_timer
        prev_scared = [game_state.get_agent_state(i) for i in self.get_opponents(successor)][0].scared_timer
        if new_scared > prev_scared:
            reward = 5

        # Pacman has died (returned to its main position)
        vis_ghosts = [a for a in cur_enemies if not a.is_pacman and a.get_position() is not None]
        if len(vis_ghosts) > 0:
            if min([self.get_maze_distance(cur_pos, g.get_position()) for g in vis_ghosts]) and next_pos == self.start:
                reward = -100

        # Our agent has eaten someone when they are in ghost form
        if cur_agent_state.is_pacman and next_agent_state.is_pacman:
            now_visible = [a for a in cur_enemies if a.is_pacman and a.get_position() is not None]
            then_visible = [a for a in new_enemies if a.is_pacman and a.get_position() is not None]
            if len(now_visible) > 0:
                if min([self.get_maze_distance(cur_pos, a.get_position()) for a in
                        now_visible]) == 1 and then_visible < now_visible:
                    return 5

        # Penalize pacman for being too close to the starting position (consider 5 steps as being too close) -> we want the attacker to get out of his territory ASAP
        next_pos = successor.get_agent_position(self.index)
        dist = self.get_maze_distance(self.start, next_pos)
        if dist <= 5:
            return -1

        return reward




