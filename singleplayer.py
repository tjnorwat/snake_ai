from cv2 import waitKey
# from SnakeGame import Snake
# from newSnakeGame import Snake, Actions
from gameSnake import Snake, Actions
# from improvedSnakeGame import Snake, Actions

player = Snake(size=6, player=True, time_between_moves=1)
player.reset()
player.render(renderer=100)

while True:
    key_press = waitKey(0)
    if key_press == ord('a'):
        action = Actions.LEFT
    elif key_press == ord('d'):
        action = Actions.RIGHT
    elif key_press == ord('w'):
        action = Actions.UP
    elif key_press == ord('s'):
        action = Actions.DOWN
    
    obs, rewards, dones, info = player.step(action)
    player.render(renderer=100)
    
    # print('obs len', len(obs))
    # print('rewards ', rewards)
    
    if dones:
        break