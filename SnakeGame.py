import cv2
import random
import numpy as np
from gym import spaces, Env


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
        # self.observation_space = spaces.Box(low=0, high=255, shape=(size * size * 3,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 3,), dtype=np.uint8)
        self.reset()


    def _GetApplePosition(self):
        choices = list()
        # making sure that the new apple doesnt spawn in the snake
        for w in self.whole_coord:
            if w not in self.snake_positions:
                choices.append(w)

        self.apple_position = random.choice(choices)
        
    
    def _GetRandomSnakePosition(self):
        self.snake_positions = [[random.randrange(0, self.size), random.randrange(0, self.size)]]
        
        i = 1
        snake_length = random.randint(3, self.size)

        while len(self.snake_positions) != snake_length:
            new_coord = [random.randrange(0, self.size), random.randrange(0, self.size)]
            
            # checking to make sure we didnt generate a coordinate that is already there 
            if new_coord not in self.snake_positions:
                # check to see if we are adjacent 
                last_pos = self.snake_positions[-1]
                if new_coord[0] == last_pos[0] and abs(new_coord[1] - last_pos[1]) == 1:
                    self.snake_positions.append(new_coord)
                elif new_coord[1] == last_pos[1] and abs(new_coord[0] - last_pos[0]) == 1:
                    self.snake_positions.append(new_coord)

            # make sure that we try and not get caught in a endless loop 
            i += 1
            if i > 500:
                i = 0
                snake_length = random.randint(1, self.size)
                self.snake_positions = [[random.randrange(0, self.size), random.randrange(0, self.size)]]


    def _GetOBSImg(self):
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

        return img


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

        # left, right, up, down
        snake_head = np.copy(self.snake_positions[0])

        if action == 0:
            snake_head[0] -= 1
        elif action == 1:
            snake_head[0] += 1
        elif action == 2:
            snake_head[1] -= 1
        else:
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
                # reward = (self.size ** 2) * .1
                # reward = 5
                # reward = 1
                # reward += (self.size ** 4 - self.total_moves) * .005

                completion_reward = 5
                # moves_reward = 1 - ( self.total_moves / self.size ** 4 )
                # reward = completion_reward * moves_reward
                
                reward = completion_reward

                self.done = True
                info['won'] = True
                
            else:
                self._GetApplePosition()

                # reward based on length of the snake
                # reward = 1 + len(self.snake_positions) * .1
                # reward = 1
                # reward based on how many moves it takes to get the apple
                # reward += (self.size ** 2 - self.moves_to_get_apple) * .01
                # reward = 1 - (self.moves_to_get_apple / (self.size ** 2) ) ** .4
                # snake_length_reward = (1 - len(self.snake_positions) / self.size ** 2) * 10
                # reward = 10 + snake_length_reward

                # snake_length_reward = len(self.snake_positions) * .006
                # reward = snake_length_reward + (self.size ** 2 - self.moves_to_get_apple) * .01
                reward = 1 + len(self.snake_positions) * .005

                reward += (1 - (self.moves_to_get_apple / self.size ** 2) ) * .0002

            self.moves_to_get_apple = 0

        # move the snake 
        else:
            self.snake_positions.insert(0, snake_head)
            self.snake_positions.pop()
            # reward = 0

            curr_distance = np.linalg.norm(np.array(snake_head) - np.array(self.apple_position))
            
            # getting closer to the apple
            if self.prev_distance != 0:
                # num_moves_reward = 1 - (self.moves_to_get_apple / (self.size ** 2))
                distance_reward = (1 - (curr_distance / self.max_distance) ) * .00005
                reward = distance_reward
            else:
                reward = 0

            self.prev_distance = curr_distance
            

        # if snakes collides with itself or the wall we die :(
        if snake_head in self.snake_positions[1:] or snake_head[0] >= self.size or snake_head[0] < 0 or snake_head[1] >= self.size or snake_head[1] < 0:
            self.done = True
            self.prev_distance = 0
            reward = -1

        # snake shouldnt loop around the apple  
        elif self.moves_to_get_apple > self.size ** 2:
            self.done = True
            self.prev_distance = 0
            reward = -.5
      
        obs = self._GetOBSImg()

        return obs, reward, self.done, info


    def reset(self):

        # self._GetRandomSnakePosition()
        # self.snake_positions = [[self.size // 2, self.size // 2], [self.size // 2, self.size // 2 - 1], [self.size // 2, self.size // 2 - 2]]
        self.snake_positions = [[self.size // 2, self.size // 2]]
        self._GetApplePosition()
        
        self.score = 0
        self.prev_distance = 0
        self.moves_to_get_apple = 0
        self.done = False

        self.total_moves = 0

        obs = self._GetOBSImg()

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