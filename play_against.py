from cv2 import waitKey
from SnakeGame import Snake
from newSnakeGame import Snake as new_snake, Actions
from newerSnake import Snake as older_snake
from gameSnake import Snake as game_snake
from render import GetNewestModel


def PlayervsAI(player1, player2, model):
    player1_done = False
    player2_done = False

    player1.reset()
    obs2 = player2.reset()

    player1.render()
    player2.render()
    while True:

        if not player1_done:

            key_press = waitKey(0)
            if key_press == ord('a'):
                action = Actions.LEFT
            elif key_press == ord('d'):
                action = Actions.RIGHT
            elif key_press == ord('w'):
                action = Actions.UP
            elif key_press == ord('s'):
                action = Actions.DOWN

            obs1, rewards1, dones1, info1 = player1.step(action)
            player1.render(renderer=100)
            if dones1:
                print('You are DONE')
                player1_done = True

        if not player2_done:
            # making the ai play the game 
            action2, _states = model.predict(obs2, deterministic=True)
            obs2, rewards2, dones2, info2 = player2.step(action2)
            player2.render(renderer=100)
            if dones2:
                print('AI DONE')
                player2_done = True
        
        if player1_done and player2_done:
            break


def AIvsAI(player1, player2, model1, model2):
    player1_done = False
    player2_done = False

    obs1 = player1.reset()
    obs2 = player2.reset()

    player1.render()
    player2.render()
    while True:

        if not player1_done:
            # making the ai play the game 
            action1, _states = model1.predict(obs1, deterministic=True)
            obs1, rewards1, dones1, info1 = player1.step(action1)
            player1.render(renderer=100)
            if dones1:
                print('PLAYER 1 DONE')
                player1_done = True

        if not player2_done:
            # making the ai play the game 
            action2, _states = model2.predict(obs2, deterministic=True)
            obs2, rewards2, dones2, info2 = player2.step(action2)
            player2.render(renderer=100)
            if dones2:
                print('PLAYER 2 DONE')
                player2_done = True
        
        if player1_done and player2_done:
            break


def WhoisBetter(player1, player2, model1, model2, total_games):

    num_games = 0
    player1_num_wins = 0
    player2_num_wins = 0

    while num_games < total_games:
        player1_done = False
        player2_done = False

        obs1 = player1.reset()
        obs2 = player2.reset()

        while True:
            if not player1_done:
                # making the ai play the game 
                action1, _states = model1.predict(obs1, deterministic=True)
                obs1, rewards1, dones1, info1 = player1.step(action1)
                # player1.render(renderer=100)
                if dones1:
                    # print('PLAYER 1 DONE')
                    player1_done = True


            if not player2_done:
                # making the ai play the game 
                action2, _states = model2.predict(obs2, deterministic=True)
                obs2, rewards2, dones2, info2 = player2.step(action2)
                # player2.render(renderer=100)
                if dones2:
                    # print('PLAYER 2 DONE')
                    player2_done = True


            # if both players finish at the same time
            if info1['won'] and info2['won']:
                break

            # if player 1 finishes first, award the win
            elif info1['won']:
                num_games += 1
                player1_num_wins += 1
                break

            # if player 2 finished first, award the win
            elif info2['won']:
                num_games += 1
                player2_num_wins += 1
                break
            
            # if either die and they havent won, restart the test ( trying to figure out who has the faster completion; dont care about deaths)
            elif player1_done or player2_done:
                break

    # win percentage calculation
    print(f'player 1 {player1.timestep} win percentage: {player1_num_wins / num_games * 100:.2f} %')
    print(f'player 2 {player2.timestep} win percentage: {player2_num_wins / num_games * 100:.2f} %')


if __name__ == '__main__':

    # recent_timestep =  1662781705 
    # recent_timestep = 1662936567 # size 4
    # recent_timestep = 1663013938
    # recent_timestep = 1662864800 
    # recent_timestep = 1665814474 
    # recent_timestep = 1665965302
    recent_timestep = 1666166799
    
    player1 = game_snake(size=6, player=True, time_between_moves=1)
    player2 = game_snake(size=6, player=False, time_between_moves=100, timestep=recent_timestep)

    model = GetNewestModel(env=player2, recent_timestep=recent_timestep, recent_file=90060000)
    # model = GetNewestModel(env=player2, recent_timestep=recent_timestep, recent_file=78165000)
    PlayervsAI(player1, player2, model)



    # # recent_timestep1 = 1662781705 
    # # recent_timestep1 = 1663013938
    # # recent_timestep1 =  1662919438 
    # # recent_timestep1 = 1665814474
    # # recent_timestep1 = 1665900538
    # # recent_timestep1 = 1665965302
    # recent_timestep1 = 1666166799


    # # recent_timestep2 = 1662864800 
    # # recent_timestep2 = 1662781705
    # recent_timestep2 = 1665814474
    # # recent_timestep2 = 1666166799

    # player1 = game_snake(size=6, player=False, time_between_moves=100, timestep=recent_timestep1)
    # player2 = older_snake(size=6, player=False, time_between_moves=100, timestep=recent_timestep2)
    # # player2 = game_snake(size=6, player=False, time_between_moves=100, timestep=recent_timestep2)

    # model1 = GetNewestModel(env=player1, recent_timestep=recent_timestep1, recent_file=90060000)
    # model2 = GetNewestModel(env=player2, recent_timestep=recent_timestep2, recent_file=78165000)


    # # AIvsAI(player1, player2, model1, model2)
    # WhoisBetter(player1, player2, model1, model2, 100)



    # recent_timestep1 =  1662781705 
    # recent_timestep1 =  1662919438 
    # # recent_timestep1 =  1663013938 

    # recent_timestep2 = 1662864800 # best one so far

    # player1 = Snake(size=6, player=False, timestep=recent_timestep1)
    # player2 = Snake(size=6, player=False, timestep=recent_timestep2)

    # model1 = GetNewestModel(env=player1, recent_timestep=recent_timestep1)
    # model2 = GetNewestModel(env=player2, recent_timestep=recent_timestep2)

    # WhoisBetter(player1, player2, model1, model2, 100)
