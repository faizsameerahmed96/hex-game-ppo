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
- [ ] Train and save again radom agent
- [ ] Train against smart agent from pa4
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