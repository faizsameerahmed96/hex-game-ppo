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
- [ ] Resume Learning every time
- [ ] Add graphs of performance


## Possible Implementations

### 2 agents that are trained on being player_1 and player_2
- Possible thing: since we are always guaranteed a winner in game of hex,
 start with training `player_1` agent until it reaches a certain `win` condition.
 After this train the other `player`. Continue doing this. This will make both the agents become very good before moving on.
 We may need to join these networks in some way if only one agent is allowed.

TO ADD


## Scratch Notes
- A higher learning rate prioritizes future learning. Explore.
- Analyze how PPO and DQN play as that will be the most common strategies.
- Investigate when pie rule is more effective