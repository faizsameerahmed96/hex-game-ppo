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
- [ ] Understand the codebase and clean
  - [x] Check if negative reward is happening
  - [ ] Go through PPO one more time
  - [ ] Clean codebase
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


## Scratch Notes
- A higher learning rate prioritizes future learning. Explore.
- Analyze how PPO and DQN play as that will be the most common strategies.
- Investigate when pie rule is more effective
- Create a distribution map of where the actions where chosen mainly in the last map
- Adaptive temperature based on win %