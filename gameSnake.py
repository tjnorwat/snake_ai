import cv2
import random
import numpy as np
from gym import spaces, Env
from collections import deque

class Actions():
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Snake(Env):
    def __init__(self, size=6, player=False, time_between_moves=100, timestep=None, threshold=-1):

        super(Snake, self).__init__()
        
        # left/right/up
        # dont need down, using it just for player
        self.direction_arr = [
            [1, -1, -1, 1],
            [-1, 1, 1, -1],
            [-1, 1, -1, 1],
            [1, -1, 1, -1]
        ]

        # x = 0, y = 1
        self.axis_arr = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]

        
        self.apple_state_threshold = threshold
        self.size = size
        self.player = player
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        self.max_distance = np.linalg.norm(np.array([0,0]) - np.array([size-1, size-1]))
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()
        # only need left/right/up because of local direction 
        self.action_space = spaces.Discrete(3)
        # snake head x y, snake tail position x y, apple position (x, y), distance to apple (dx, dy), number of moves taken, snake length, distance to tail (dx, dy), is square open left/right/up, direction
        extra_obs_num = 16
        # max coordinate pairs of snake 
        # shape_size = size ** 2 * 2
        # previous action length
        shape_size = size ** 2
        # adding extra obs 
        shape_size += extra_obs_num
        # 'high' parameter large size comes from max number which would be number of moves taken
        self.observation_space = spaces.Box(low=-size, high=size**2, shape=(shape_size,), dtype=np.int8)


    def _GetApplePosition(self):
        # making sure that the new apple doesnt spawn in the snake

        if self.apple_state_threshold > random.random():
            head = self.snake_positions[0]
            # check to make sure we are in bounds and we are not in the snake 
            left_choice = head[0] - 1
            right_choice = head[0] + 1
            up_choice = head[1] - 1
            down_choice = head[1] + 1

            choices = list()

            if left_choice >= 0:
                temp = [left_choice, head[1]]
                if temp not in self.snake_positions:
                    choices.append(temp)

            if right_choice < self.size:
                temp = [right_choice, head[1]]
                if temp not in self.snake_positions:
                    choices.append(temp)

            if up_choice >= 0:
                temp = [head[0], up_choice]
                if temp not in self.snake_positions:
                    choices.append(temp)
            
            if down_choice < self.size:
                temp = [head[0], down_choice]
                if temp not in self.snake_positions:
                    choices.append(temp)

            # sometimes we dont have space available for the apple, so we need to get a random position
            if len(choices) != 0:
                # have a 10% chance min to have an apple spawn next to snake
                if self.apple_state_threshold > .1:
                    # 70 million times ? 
                    # should change to whole game instead of per instance of eating apple 
                    self.apple_state_threshold -= .0000001
                self.apple_position = random.choice(choices)
                return

        choices = [choice for choice in self.whole_coord if choice not in self.snake_positions]
        self.apple_position = random.choice(choices)

    
    def _GetOBS(self):

        # if the length of the snake is just starting out, we need to set the tail to the head 
        if len(self.snake_positions) == 1:
            snake_tail = self.snake_positions[0]
            distance_to_tail_x = 0
            distance_to_tail_y = 0
        else:
            snake_tail = self.snake_positions[-1]
            # distance_to_tail = round(np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.snake_positions[-1])), 2)
            distance_to_tail_x = self.snake_positions[-1][0] - self.snake_positions[0][0]
            distance_to_tail_y = self.snake_positions[-1][1] - self.snake_positions[0][1]

        # distance_to_apple = round(np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.apple_position)), 2)
        distance_to_apple_x = self.apple_position[0] - self.snake_positions[0][0]
        distance_to_apple_y = self.apple_position[1] - self.snake_positions[0][1]


        # make sure that we are in bounds of the game
        # make sure that we dont count another part of the snake
            # if the part of the snake is the tail, we count it 
        # if the square is the apple, we count it  

        # is left/right/up square open ?
        # down will always be not available since its local direction
        open_squares = [0, 0, 0]
        snake_head = [ self.snake_positions[0][0], self.snake_positions[0][1] ]

        for i in range(3):
            
            val = self.direction_arr[i][self.direction]
            which_axis = self.axis_arr[i][self.direction]

            if which_axis == 0:
                # if we are in bounds
                if snake_head[0] + val >= 0 and snake_head[0] + val < self.size:
                    # check if the head is against another snake part but not the tail
                    if [ snake_head[0] + val, snake_head[1] ] not in self.snake_positions[:-1]:
                        open_squares[i] = 1
            else:
                if snake_head[1] + val >= 0 and snake_head[1] + val < self.size:
                    if [ snake_head[0], snake_head[1] + val ] not in self.snake_positions[:-1]:
                        open_squares[i] = 1

        obs = snake_head + [
            snake_tail[0],
            snake_tail[1],
            self.apple_position[0],
            self.apple_position[1],
            self.moves_to_get_apple,
            len(self.snake_positions),
            distance_to_apple_x,
            distance_to_apple_y,
            distance_to_tail_x,
            distance_to_tail_y,
            self.direction
            ] \
                + open_squares \
                + list(self.prev_actions)
        
        # print(obs)

        return np.array(obs)


    def _GetRenderImg(self, renderer=100):
        img = np.zeros((self.size* renderer, self.size * renderer, 3), dtype=np.uint8)

        # drawing the snake
        head = self.snake_positions[0]
        cv2.rectangle(img=img, pt1=(head[0] * renderer, head[1] * renderer), pt2=(head[0] * renderer + renderer, head[1] * renderer + renderer), color=(255,0,0), thickness=-1)

        for position in self.snake_positions[1:-1]:
            cv2.rectangle(img=img, pt1=(position[0] * renderer, position[1] * renderer), pt2=(position[0] * renderer + renderer, position[1] * renderer + renderer), color=(0,255,0), thickness=-1)

        if len(self.snake_positions) > 1:
            tail = self.snake_positions[-1]
            cv2.rectangle(img=img, pt1=(tail[0] * renderer, tail[1] * renderer), pt2=(tail[0] * renderer + renderer, tail[1] * renderer + renderer), color=(255,255,255), thickness=-1)

        # drawing the apple
        cv2.rectangle(img=img, pt1=(self.apple_position[0] * renderer, self.apple_position[1] * renderer), pt2=(self.apple_position[0] * renderer + renderer, self.apple_position[1] * renderer + renderer), color=(0,0,255), thickness=-1)

        return img


    def step(self, action):

        info = {
            'won' : False
        }

        # snake_head = np.copy(self.snake_positions[0])
        snake_head = [ self.snake_positions[0][0], self.snake_positions[0][1] ]
        self.prev_actions.append(action)

        # if player is not human 
        if not self.player:

            val = self.direction_arr[action][self.direction]
            which_axis = self.axis_arr[action][self.direction]

            # get new direction
            if val == -1 and which_axis == 0:
                self.direction = Actions.LEFT
            elif val == -1 and which_axis == 1:
                self.direction = Actions.UP
            elif val == 1 and which_axis == 0:
                self.direction = Actions.RIGHT
            else:
                self.direction = Actions.DOWN
            
            # move the snake head
            if which_axis == 0:
                snake_head[0] += val
            else:
                snake_head[1] += val

        # if player is human we switch to more intuitive controls
        else:

            # left, right, up, down
            if action == Actions.LEFT:
                snake_head[0] -= 1
            elif action == Actions.RIGHT:
                snake_head[0] += 1
            elif action == Actions.UP:
                snake_head[1] -= 1
            elif action == Actions.DOWN:
                snake_head[1] += 1

        self.moves_to_get_apple += 1
        self.total_moves += 1
        
        # increase snake length of eating the apple
        if snake_head == self.apple_position:
            self.score += 1
            self.snake_positions.insert(0, snake_head)

            # we completed the game 
            if len(self.snake_positions) == self.size ** 2:
                reward = 2
                print('total_moves', self.total_moves)
                self.done = True
                info['won'] = True

            # we ate an apple 
            else:
                self._GetApplePosition()
                reward = (1 - (self.moves_to_get_apple / (self.size ** 2 - 1))) * .5
            
            # resetting the previous distance to new position of apple
            self.prev_distance = np.linalg.norm(np.array(snake_head) - np.array(self.apple_position))
            # resetting the counter after we eat an apple 
            self.moves_to_get_apple = 0

        # move the snake 
        else:
            self.snake_positions.insert(0, snake_head)
            self.snake_positions.pop()

            curr_distance = np.linalg.norm(np.array(snake_head) - np.array(self.apple_position))
            reward = (1 - (curr_distance / self.max_distance) ) * .002

            self.prev_distance = curr_distance
            

        # if snakes collides with itself or the wall we die :(
        if snake_head in self.snake_positions[1:] or snake_head[0] >= self.size or snake_head[0] < 0 or snake_head[1] >= self.size or snake_head[1] < 0:
            self.done = True
            reward = -1

        # snake shouldnt loop around the apple  
        elif self.moves_to_get_apple >= self.size ** 2:
            self.done = True
            reward = -.5

        return self._GetOBS(), reward, self.done, info


    def reset(self):

        self.snake_positions = [[self.size // 2, self.size // 2]]
        self._GetApplePosition()
        self.score = 0

        self.prev_actions = deque(maxlen=self.size ** 2)
        for _ in range(self.size ** 2):
            self.prev_actions.append(-1)

        # updated to get rewards correctly 
        self.prev_distance = np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.apple_position))
        self.moves_to_get_apple = 0
        self.done = False

        self.total_moves = 0

        # snake is facing up 
        self.direction = 2

        obs = self._GetOBS()

        return obs


    def render(self, renderer=100):
        img = self._GetRenderImg(renderer=renderer)
        if self.player:
            cv2.imshow('Player', img)
        elif self.timestep:
            cv2.imshow(f'Snake AI {self.timestep}', img)
        else:
            cv2.imshow('Snake AI', img)
        
        cv2.waitKey(self.time_between_moves)