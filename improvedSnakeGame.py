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
    def __init__(self, size=6, player=False, time_between_moves=100, timestep=None):

        super(Snake, self).__init__()

        self.size = size
        self.player = player
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        self.max_distance = np.linalg.norm(np.array([0,0]) - np.array([size-1, size-1]))

        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()
        self.action_space = spaces.Discrete(4)
        # distance to apple, moves to get apple, distance to tail, length of snake
        extra_obs = 4
        # previous moves
        extra_obs += size ** 2
        # num moves to get apple or pixel rgb
        high = size ** 2 + 1 if size ** 2 + 1 > 255 else 255
        self.observation_space = spaces.Box(low=-1, high=high, shape=(size ** 2 * 3 + extra_obs,), dtype=np.uint8)
        # self.reset()


    def _GetApplePosition(self):
        # making sure that the new apple doesnt spawn in the snake
        choices = [choice for choice in self.whole_coord if choice not in self.snake_positions]
        self.apple_position = random.choice(choices)
        

    def _GetOBS(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
       # drawing the snake
        head = self.snake_positions[0]
        cv2.rectangle(img=img, pt1=(head[0], head[1]), pt2=(head[0], head[1]), color=(255,0,0), thickness=-1)

        for position in self.snake_positions[1:-1]:
            cv2.rectangle(img=img, pt1=(position[0], position[1]), pt2=(position[0], position[1]), color=(0,255,0), thickness=-1)

        if len(self.snake_positions) > 1:
            tail = self.snake_positions[-1]
            cv2.rectangle(img=img, pt1=(tail[0], tail[1]), pt2=(tail[0], tail[1]), color=(255,255,255), thickness=-1)

        # drawing the apple
        cv2.rectangle(img=img, pt1=(self.apple_position[0], self.apple_position[1]), pt2=(self.apple_position[0], self.apple_position[1]), color=(0,0,255), thickness=-1)

        img = img.flatten()


        if len(self.snake_positions) == 1:
            distance_to_tail = 0
        else:
            distance_to_tail = round(np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.snake_positions[-1])), 2)

        distance_to_apple = round(np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.apple_position)), 2)

        img = np.append(img, [
            distance_to_apple,
            self.moves_to_get_apple,
            distance_to_tail,
            len(self.snake_positions)
        ])

        img = np.append(img, list(self.prev_actions))

        return img


    def _GetRenderImg(self, renderer=100):
        img = np.zeros((self.size * renderer, self.size * renderer, 3), dtype=np.uint8)

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

        snake_head = np.copy(self.snake_positions[0])
        self.prev_actions.append(action)

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
        snake_head = snake_head.tolist()
        self.total_moves += 1
        # increase snake length of eating the apple
        if snake_head == self.apple_position:
            self.prev_distance = 0
            self.score += 1
            
            self.snake_positions.insert(0, snake_head)

            # we completed the game 
            if len(self.snake_positions) == self.size ** 2:
                reward = 5

                self.done = True
                info['won'] = True
                
            else:
                self._GetApplePosition()
                reward = (1 - (self.moves_to_get_apple / self.size ** 2) )
            
            # resetting the previous distance 
            self.prev_distance = np.linalg.norm(np.array(snake_head) - np.array(self.apple_position))
            # resetting the counter after we eat an apple 
            self.moves_to_get_apple = 0

        # move the snake 
        else:
            self.snake_positions.insert(0, snake_head)
            self.snake_positions.pop()
            # reward = 0

            curr_distance = np.linalg.norm(np.array(snake_head) - np.array(self.apple_position))
            
            # getting closer to the apple
            if self.prev_distance != 0:
                distance_reward = (1 - (curr_distance / self.max_distance) ) * .004
                reward = distance_reward
            else:
                reward = 0

            self.prev_distance = curr_distance
            

        # if snakes collides with itself or the wall we die :(
        if snake_head in self.snake_positions[1:] or snake_head[0] >= self.size or snake_head[0] < 0 or snake_head[1] >= self.size or snake_head[1] < 0:
            self.done = True
            # self.prev_distance = 0
            reward = -1

        # snake shouldnt loop around the apple  
        elif self.moves_to_get_apple >= self.size ** 2:
            self.done = True
            # self.prev_distance = 0
            reward = -.5
      
        obs = self._GetOBS()

        return obs, reward, self.done, info


    def reset(self):
        
        self.snake_positions = [[self.size // 2, self.size // 2]]
        self._GetApplePosition()
        
        self.prev_actions = deque(maxlen=self.size ** 2)
        for _ in range(self.size ** 2):
            self.prev_actions.append(-1)

        self.score = 0
        # updated to get rewards correctly 
        self.prev_distance = np.linalg.norm(np.array(self.snake_positions[0]) - np.array(self.apple_position))
        self.moves_to_get_apple = 0
        self.done = False

        self.total_moves = 0

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