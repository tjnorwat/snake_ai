new snake game is supposed to not use pixels for the observation space 
snakegame just uses pixels for the obs space
improvedsnakegame uses pixels plus extra observations 

# Observation Space
The observation space can vary. I originally used the pixels but am testing a custom observation that includes:
* the snake positions (x, y) with zeros filled in for max length. This space has less parameters than using the pixels.
* apple position (x, y)
* moves taken to get apple
* length of snake
* distance to apple
* distance to tail
* previous actions 
* direction the snake is facing? 
* how far each direction snake can go (num open squares left/right/forward)

# Action Space
Keeping it pretty simple with just 
* Left
* Right
* Up
* Down

# Best models
These are some of the best models that I have come up with, with the best being at the top. All of these are size 6 so far

1. `1666166799` (gamesnake obs, uses local direction)
2. `1665965302` (can sometimes be the best, uses newsnake)
3. `1665814474` (newer snake observation)
4. `1665900538` (newer snake observation with binary open squares)
5. `1663013938`
6. `1662864800`
7. `1662919438`
8. `1662781705`


# Having the AI play against other AI
There are 3 different ways that the AI can play the game after training:
* Against a human
* Against another model (or itself) in a slow manner where you can see the individual moves
* Against another model but plays a lot of games (this is more for testing to see which AI can complete the game faster)

# TODO 
Different things that i would like to implement 
* different obs 
  * snake positions  
  * apple position
  * snake head distance from apple 
  * number of moves taken for each apple 
  * snake length 
  * distance to tail
  * 
* random apple available positions are updated every move ?
* learning function is static rather than based on snake length  
* pause/resume training
* after getting the apple, the distance doesnt work correctly ( works now)
* auto size renderer

* change to pygame
* make sure the snake can't go backwards on itself
* curriculum learning? / random states 
  * difference between having the food spawn in an NxN, leaving the whole grid open; vs playing in an NxN grid
* switch to local movement
  * redo which square is open to be local
* reward function to be based on total num moves? take longer trianing time 