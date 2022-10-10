import cv2
import time
import random
import numpy as np
from gym import spaces, Env

SIZE = 6
renderer = 100
score = 0
done = False


def randomsnakeposition():
    snake_positions = [[random.randrange(0, SIZE), random.randrange(0, SIZE)]]
    snake_length = random.randint(1,SIZE)
    i = 1
    while len(snake_positions) != snake_length:
        new_coord = [random.randrange(0, SIZE), random.randrange(0, SIZE)]
        
        # checking to make sure we didnt generate a coordinate that is already there 
        if new_coord not in snake_positions:
            # check to see if we are adjacent 
            last_pos = snake_positions[-1]
            if new_coord[0] == last_pos[0] and abs(new_coord[1] - last_pos[1]) == 1:
                snake_positions.append(new_coord)
            elif new_coord[1] == last_pos[1] and abs(new_coord[0] - last_pos[0]) == 1:
                snake_positions.append(new_coord)
        i += 1
        if i > 500:
            i = 0
            snake_positions = [[random.randrange(0, SIZE), random.randrange(0, SIZE)]]


    return snake_positions


snake_positions = randomsnakeposition()
# def _GetApplePosition():
#     while True:
#         apple_position = [random.randrange(0, SIZE), random.randrange(0, SIZE)]
#         if apple_position not in snake_positions:
#             return apple_position


whole_coord = np.mgrid[0:SIZE, 0:SIZE].reshape(2, -1).T.tolist()

def _GetApplePosition():

    apple = list()
    for w in whole_coord:
        if w not in snake_positions:
            apple.append(w)

    return random.choice(apple)

    # randomly select 


def _GetRenderImg(snake_positions, apple_position, renderer=100):
    img = np.zeros((SIZE* renderer, SIZE * renderer, 3), dtype=np.uint8)
    
    # drawing the snake
    head = snake_positions[0]
    cv2.rectangle(img=img, pt1=(head[0] * renderer, head[1]* renderer), pt2=(head[0] * renderer + renderer, head[1] * renderer + renderer), color=(255,0,0), thickness=-1)
    for position in snake_positions[1:]:
        cv2.rectangle(img=img, pt1=(position[0] * renderer, position[1]* renderer), pt2=(position[0] * renderer + renderer, position[1] * renderer + renderer), color=(0,255,0), thickness=-1)

    # drawing the apple
    cv2.rectangle(img=img, pt1=(apple_position[0] * renderer, apple_position[1] * renderer), pt2=(apple_position[0] * renderer + renderer, apple_position[1] * renderer + renderer), color=(0,0,255), thickness=-100)

    return img



def _GetOBSImg(snake_positions, apple_position):
    img = np.zeros((SIZE, SIZE , 3), dtype=np.uint8)
    
    # drawing the snake
    head = snake_positions[0]
    cv2.rectangle(img=img, pt1=(head[0] , head[1]), pt2=(head[0]  , head[1]  ), color=(255,0,0), thickness=-1)
    for position in snake_positions[1:]:
        cv2.rectangle(img=img, pt1=(position[0] , position[1]), pt2=(position[0]  , position[1]  ), color=(0,255,0), thickness=-1)

    # drawing the apple
    cv2.rectangle(img=img, pt1=(apple_position[0] , apple_position[1] ), pt2=(apple_position[0]  , apple_position[1]  ), color=(0,0,255), thickness=-100)

    return img

apple_position = _GetApplePosition()


other_img = _GetOBSImg(snake_positions, apple_position)
img = _GetRenderImg(snake_positions, apple_position)

img = _GetRenderImg(snake_positions, apple_position)

cv2.imshow('YO', img)
cv2.waitKey(1)

while True:
    # left, right, up, down
    snake_head = np.copy(snake_positions[0])

    action = cv2.waitKey(0)

    if action == ord('a'):
        snake_head[0] -= 1
    elif action == ord('d'):
        snake_head[0] += 1
    elif action == ord('w'):
        snake_head[1] -= 1
    else:
        snake_head[1] += 1

    snake_head = snake_head.tolist()
    # increase snake length of eating the apple
    if snake_head == apple_position:
        prev_distance = 0
        score += 1
        
        snake_positions.insert(0, snake_head)
        apple_position = _GetApplePosition()
        reward = 1

    # move the snake 
    else:
        snake_positions.insert(0, snake_head)
        snake_positions.pop()

    if snake_head in snake_positions[1:] or snake_head[0] >= SIZE or snake_head[0] < 0 or snake_head[1] >= SIZE or snake_head[1] < 0:
        done = True
        prev_distance = 0
        reward = -1
        print("DONE")
    

    img = _GetRenderImg(snake_positions, apple_position)
    cv2.imshow('YO', img)
    cv2.waitKey(1)