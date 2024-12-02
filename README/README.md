# PA5 Analysis

## Research Phase
- [x] Go through all the slides in brief and do some research
- [x] Explore Game of Hex environment.
- [x] Create a basic PPO example for dumb agent.
- [x] Add rendering after every some time
- [x] Only use full episodes
- [x] Create a system for saving
- [x] Check if player 2 works fine
- [x] Create a double learning agent
- [x] Understand the codebase and clean
  - [x] Check if negative reward is happening
  - [x] Go through PPO one more time
  - [x] Clean codebase
- [x] Update our hex game
- [x] Implement custom opponents
- [x] Train and save again radom agent
- [ ] Train against smart agent from pa4, maybe some distribution based on opponent player!
- [ ] Add file logging for important updates
- [ ] Keep a win replay buffer in order to retrain on things that already happened when we collapse to 0% win.
- [ ] Create a function to simulate flow between 2 agents
- [ ] Make it resumable, ability to start with player_2 training if player 1 was the last trained
- [ ] Add a way to manually specify what we want
- [ ] Ability to define custom network for training against
- [ ] Common strategies win/lose
- [ ] Make opponenet into interface and allow it to be random or whatever it wants
- [ ] Delete older agents every time we save
- [ ] Resume Learning every time (along with which model was training last)
- [ ] Evaluate a specific model against the other
- [ ] Add render slo-mo
- [ ] Add graphs of performance


## Possible Implementations

### 2 agents that are trained on being player_1 and player_2
- Possible thing: since we are always guaranteed a winner in game of hex,
 start with training `player_1` agent until it reaches a certain `win` condition.
 After this train the other `player`. Continue doing this. This will make both the agents become very good before moving on.
 We may need to join these networks in some way if only one agent is allowed.

TO ADD


## Journal
- Initially, both agents learn on random. However while dueling the agent seems to stop becoming better and settles around the 0-10% mark
- The agent takes `40` iterations and `2m` to reach `90+% win rate` against the random strategy with following config.
  ```
  "episodes_per_batch": 10,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 20,
        "lr": 3e-5,
        "clip": 0.2,
        "save_freq": 200,

        "render_every_x_iterations": 100,
  ```

### Default hyperparameters
```python
hyperparameters = {
        "episodes_per_batch": 20,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 15,
        "lr": 3e-5,
        "clip": 0.2,
        "save_freq": 200_000,

        "render_every_x_iterations": 100,

        "max_num_of_episodes_to_calculate_win_percent": 20,
        "break_after_x_continuous_win_percent": 101,
        "how_many_consecutive_wins_to_break": 5,

        "train_against_opponent": False,
        "opponent": RandomOpponent(),
    }
```


### Full Random
- We will train it again a completely random opponent initially. We use the following hyperparameters.
```
-------------------- Iteration #68 --------------------
PLAYER: player_2
Average Episodic Length: 64.35
Average Episodic Return: 22.8
Average Loss: 0.12276
Training time elapsed in min 5.258700386683146
Wins % in last 20 episodes = 95.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 134811
Iteration took: 3.39 secs
------------------------------------------------------

-------------------- Iteration #119 --------------------
PLAYER: player_1
Average Episodic Length: 80.0
Average Episodic Return: -3.6
Average Loss: 0.09049
Training time elapsed in min 9.77909373442332
Wins % in last 20 episodes = 80.0%
Wins % breakout percentage = 90.0%
Timesteps So Far: 246005
Iteration took: 3.81 secs
------------------------------------------------------
```

### Smart Random
- We will generate a random agent that used probability distributions to come up with common ways of winning

#### Center Weight 1
```
-------------------- Iteration #70 --------------------
PLAYER: player_1
Average Episodic Length: 67.6
Average Episodic Return: 26.7
Average Loss: 0.12283
Training time elapsed in min 4.627847230434417
Wins % in last 20 episodes = 100.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 108371
Iteration took: 3.34 secs
------------------------------------------------------

-------------------- Iteration #19 --------------------
PLAYER: player_2
Average Episodic Length: 52.65
Average Episodic Return: 28.65
Average Loss: 0.12927
Training time elapsed in min 1.116714564959208
Wins % in last 20 episodes = 95.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 27764
Iteration took: 2.65 secs
------------------------------------------------------
```

#### Center Weight 5 
This agent is more focused towards winning. With the default hyperparameters, it was not able to beat the smart agent. It kept reaching a breakout % of `20%` at most. This could mean the agent is not learning enough. 

Let us try the follow hyperparameters.
```json
"break_after_x_continuous_win_percent": 90,
"render_every_x_iterations": 20,
"clip": 0.3,
"lr": 5e-5,
```
```
PLAYER: player_1
Average Episodic Length: 41.6
Average Episodic Return: -80.8
Average Loss: -0.0119
Training time elapsed in min 8.746547667185466
Wins % in last 20 episodes = 0.0%
Wins % breakout percentage = 14.0%
Timesteps So Far: 190843
Iteration took: 2.26 secs
```


We are still not moving forward, let us train it for a dumber agent before moving ahead

#### Center Weight 3
No progress 

#### Center Weight 2
No progress


## Update the network to involve more convolution layers (network_2)

### Full Random
```
-------------------- Iteration #37 --------------------
PLAYER: player_1
Average Episodic Length: 77.3
Average Episodic Return: 21.85
Average Loss: 0.11401
Training time elapsed in min 3.4245606660842896
Wins % in last 20 episodes = 100.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 73880
Iteration took: 4.51 secs
------------------------------------------------------

PLAYER: player_2
Average Episodic Length: 82.2
Average Episodic Return: 19.9
Average Loss: 0.09309
Training time elapsed in min 6.68732625246048
Wins % in last 20 episodes = 100.0%
Wins % breakout percentage = 85.0%
Timesteps So Far: 143003
Iteration took: 4.6 secs
```

### Smart Random 1
```
-------------------- Iteration #14 --------------------
PLAYER: player_1
Average Episodic Length: 61.5
Average Episodic Return: 17.7
Average Loss: 0.11272
Training time elapsed in min 0.9132973829905192
Wins % in last 20 episodes = 90.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 18767
Iteration took: 3.33 secs
------------------------------------------------------
```

### Smart Random 1.5
```
hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 90,
        "render_every_x_iterations": 20,
        "episodes_per_batch": 20,
        "clip": 0.2,
        "lr": 3e-5,
        "n_updates_per_iteration": 20,
        "gamma": 0.6,
    }
```
```
-------------------- Iteration #67 --------------------
PLAYER: player_1
Average Episodic Length: 38.8
Average Episodic Return: 41.1
Average Loss: 0.111
Training time elapsed in min 4.849908149242401
Wins % in last 20 episodes = 100.0%
Wins % breakout percentage = 90.0%
Timesteps So Far: 77360
Iteration took: 3.05 secs
------------------------------------------------------

-------------------- Iteration #38 --------------------
PLAYER: player_2
Average Episodic Length: 45.0
Average Episodic Return: 38.5
Average Loss: 0.12617
Training time elapsed in min 2.6142526706059774
Wins % in last 20 episodes = 100.0%
Wins % breakout percentage = 90.0%
Timesteps So Far: 42573
Iteration took: 3.28 secs
------------------------------------------------------
```

### Smart Random 2
```
hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 90,
        "render_every_x_iterations": 20,
        "episodes_per_batch": 20,
        "clip": 0.2,
        "lr": 3e-5,
        "n_updates_per_iteration": 20,
        "gamma": 0.6,
    }
```
```
PLAYER: player_1
Average Episodic Length: 52.75
Average Episodic Return: 28.1
Average Loss: 0.10471
Training time elapsed in min 11.993719983100892
Wins % in last 20 episodes = 95.0%
Wins % breakout percentage = 89.0%
Timesteps So Far: 172864
Iteration took: 3.81 secs
------------------------------------------------------
```



## Scratch Notes
- A higher learning rate prioritizes future learning. Explore.
- Analyze how PPO and DQN play as that will be the most common strategies.
- Investigate when pie rule is more effective
- Create a distribution map of where the actions where chosen mainly in the last map
- Adaptive temperature based on win %
- A single model learning that is trained for both player 1 and player 2. This way it can form associations from what it has already learnt!
- Apply UCB alternative during exploration so that it run through all the parts of a board
- Clip the -1 reward to 0 to promote delaying the game and therefore future learning
- After a certain iteration 
  - change parameters
  - change reward structure
  - focus on won games and keep % of batch that much
- Create a buffer of batches where things went well and keep replaying it if we are stuck with win%