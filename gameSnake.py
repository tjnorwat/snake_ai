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
    def __init__(self, size=6, player=False, time_between_moves=100, timestep=None, threshold=-1, spawn_apple_stages=False):

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
        # for image
        self.min_moves = (size ** 2 - 1) ** 2
        self.spawn_apple_stages = spawn_apple_stages
        self.max_len = 1

        # dictionary of different sizes of board for apple to spawn
        if spawn_apple_stages:
            self.whole_coord = dict()
            self.weights = list()
            for weights, i in enumerate(range(6, size + 2, 2)):
                # self.whole_coord[i] = np.mgrid[0:i, 0:i].reshape(2, -1).T.tolist()
                # trying to keep the apple spawning in the middle; big brain ?? 
                self.whole_coord[i] = np.mgrid[size // 2 - i // 2:size // 2 + i // 2, size // 2 - i // 2:size // 2 + i // 2].reshape(2, -1).T.tolist()
                self.weights.append(weights + 1)
        else:
            self.whole_coord = { size:  np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist() }
        
        
        self.apple_state_threshold = threshold
        self.size = size
        self.player = player
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        self.max_distance = np.linalg.norm(np.array([0,0]) - np.array([size-1, size-1]))
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
        # print('shape_size', shape_size)
        # 'high' parameter large size comes from max number which would be number of moves taken
        self.observation_space = spaces.Box(low=-size, high=size**2, shape=(shape_size,), dtype=np.int8)


    def _GetApplePosition(self):
        # making sure that the new apple doesnt spawn in the snake
        # check length of snake first depending on apple spawn area ; for next steps 

        if self.spawn_apple_stages:
            choices = None
            # i think we can guarantee a spot because of terminal condition for finishing the game
            while True:
                choices = [choice for choice in self.whole_coord[self.apple_spawn_size] if choice not in self.snake_positions]
                if choices:
                    break
                else:
                    self.apple_spawn_size += 2

        else:
            choices = [choice for choice in self.whole_coord[self.size] if choice not in self.snake_positions]

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

        # need to add extra -1s for prev actions; just max size of snake - maxlen of prev actions
        extra_1s = [-1] * (self.size ** 2 - self.prev_actions.maxlen)

        obs = snake_head + [
            snake_tail[0],
            snake_tail[1],
            distance_to_tail_x,
            distance_to_tail_y,
            self.direction,
            self.apple_position[0],
            self.apple_position[1],
            distance_to_apple_x,
            distance_to_apple_y,
            self.moves_to_get_apple,
            len(self.snake_positions)
            ] \
                + open_squares \
                + list(self.prev_actions) \
                + extra_1s
        
        # print('prev actions', self.prev_actions)
        # print('len zeros', len(extra_1s))

        # print('obs len', len(obs))


        return np.array(obs)


    def _GetRenderImg(self, renderer=100):
        img = np.zeros(( self.size * renderer, self.size * renderer, 3), dtype=np.uint8)

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

        # for the STREAM 
        padding = np.full((self.size * renderer, 300, 3), 125, dtype=np.uint8)
        img = np.append(img, padding, axis=1)

        cv2.putText(
            img=img, 
            text=f'Length: {len(self.snake_positions)}', 
            org=( ((self.size) * renderer), renderer * 1), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Total Moves: {self.total_moves}', 
            org=( ((self.size) * renderer), renderer * 2), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Min Moves: {self.min_moves}', 
            org=( ((self.size) * renderer), renderer * 3), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Best len: {self.max_len}', 
            org=( ((self.size) * renderer), renderer * 4), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Board size: {self.size}x{self.size}', 
            org=( ((self.size) * renderer), renderer * 5), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        return img


    def step(self, action):

        info = {
            'won' : False
        }

        # snake_head = np.copy(self.snake_positions[0])
        snake_head = [ self.snake_positions[0][0], self.snake_positions[0][1] ]


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

        # self.prev_actions.append(action)
        # increase snake length of eating the apple
        if snake_head == self.apple_position:
            
            self.score += 1
            self.snake_positions.insert(0, snake_head)

            self.prev_actions = deque(list(self.prev_actions), maxlen=self.prev_actions.maxlen + 1)
            self.prev_actions.appendleft(action)

            if len(self.snake_positions) > self.max_len:
                self.max_len = len(self.snake_positions)

            # we completed the game 
            if len(self.snake_positions) == self.size ** 2:

                if self.total_moves < self.min_moves:
                    self.min_moves = self.total_moves

                reward = 5
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

            self.prev_actions.appendleft(action)

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
        
        # for getting apple position in certain area
        # making sure that if we surpass a legnth of a smaller board size, we can disregard it 
        # also we dont want to have a 0 weight for the size of the board or else we cant train :(, just for the lower parts
        if self.spawn_apple_stages:
            for weight, i in enumerate(range(6, self.size, 2)):
                if self.weights[weight] != 0:
                    if self.max_len >= i ** 2:
                        self.weights[weight] = 0

            # getting random num between 6 and lengh of board, every even number to determine where apple spawns
            self.apple_spawn_size = random.choices([i for i in range(6, self.size + 2, 2)], weights=self.weights)[0]
        # self.apple_spawn_size = 10

        self._GetApplePosition()
        self.score = 0

        # self.prev_actions = deque([-1] * self.size ** 2, maxlen=self.size ** 2)
        # trying to have maxlen be size of snake because we only care about moves in correspondance to size of snake
        self.prev_actions = deque([-1], maxlen=1)

        # updated to get rewards correctly 
        self.prev_distance = np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.apple_position))
        self.moves_to_get_apple = 0
        self.done = False

        self.total_moves = 0
        # snake is facing up 
        self.direction = 2


        return self._GetOBS()


    def render(self, renderer=100):
        img = self._GetRenderImg(renderer=renderer)
        if self.player:
            cv2.imshow('Player', img)
        elif self.timestep:
            cv2.imshow(f'Snake AI {self.timestep}', img)
        else:
            cv2.imshow('Snake AI', img)
        
        cv2.waitKey(self.time_between_moves)
