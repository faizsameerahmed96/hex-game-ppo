import os
import random
import time
from collections import deque
from queue import Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ourhexgame.ourhexenv import OurHexGame


class PPO:

    def __init__(
        self,
        policy_class,
        env: OurHexGame,
        current_agent_player="player_2",
        **hyperparameters,
    ):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                current_agent_player - are we player 1 or player 2
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
                None
        """
        self.episodes_per_batch = hyperparameters.get("episodes_per_batch", 10)
        self.max_timesteps_per_episode = hyperparameters.get(
            "max_timesteps_per_episode", 1600
        )
        self.render_every_x_iterations = hyperparameters.get(
            "render_every_x_iterations", 5
        )
        self.n_updates_per_iteration = hyperparameters.get("n_updates_per_iteration", 5)
        self.lr = hyperparameters.get("lr", 0.005)
        self.gamma = hyperparameters.get("gamma", 0.95)
        self.clip = hyperparameters.get("clip", 0.17)
        self.save_freq = hyperparameters.get("save_freq", 10)
        self.step_reward_multiplier = hyperparameters.get("step_reward_multiplier", 1)

        self.break_after_x_continuous_win_percent = hyperparameters.get(
            "break_after_x_continuous_win_percent", 95
        )
        self.how_many_consecutive_wins_to_break = hyperparameters.get(
            "how_many_consecutive_wins_to_break", 1
        )
        self.max_num_of_episodes_to_calculate_win_percent = hyperparameters.get(
            "max_num_of_episodes_to_calculate_win_percent", 20
        )
        self.wins_queue = deque(
            maxlen=self.max_num_of_episodes_to_calculate_win_percent
        )
        self.wins_percent_queue = deque(maxlen=self.how_many_consecutive_wins_to_break)

        self.seed = hyperparameters.get("seed", None)

        self.opponent = hyperparameters.get("opponent", None)

        if self.opponent is None:
            raise Exception("Opponent is not defined")

        if self.seed != None:
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

        self.env = env
        self.current_agent_player = current_agent_player
        self.other_agent_player = (
            "player_1" if current_agent_player == "player_2" else "player_2"
        )

        self.start_learn_time = time.time()

        self.act_dim = env.action_space(current_agent_player).n

        # To select the next action (theta_k)
        self.actor = policy_class(env.board_size, self.act_dim)
        # To estimate the value function (V_phi)
        self.critic = policy_class(env.board_size, 1)

        # The opponent actor we want to fight
        self.opponent_actor = policy_class(env.board_size, self.act_dim)
        self.opponent_actor.eval()

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def load_model_for(self, agent: str, model_path: str):
        """
        Load a model for a specific agent

        Parameters:
            agent - the agent to load the model for
            model_path - the path to the model

        Returns:
            None
        """
        print(f"Loading model from {model_path}")
        self.actor.load_state_dict(torch.load(f"{model_path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{model_path}/critic.pth"))


    def transform_observation(self, obs):
        """
        Convert observation into a one hot encoded tensor for conv2d layers

        Parameters:
            obs - the observation dictionary

        Returns:
            Tensor - a flattened tensor representation of the observation
        """
        board = obs["observation"]
        one_hot_board = np.eye(3)[board]  # one hot encode the board
        one_hot_board = np.transpose(
            one_hot_board, (2, 0, 1)
        )  # transpose to (channels, height, width)
        return torch.tensor(one_hot_board, dtype=torch.float32)

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
                total_timesteps - the total number of timesteps to train for. Pass - 1 to train indefinitely.
        """
        print(f"Learning for {total_timesteps} timesteps")

        self.start_learn_time = time.time()

        timesteps_so_far = 0
        iterations_so_far = 0  # Iterations ran so far

        while timesteps_so_far < total_timesteps or total_timesteps == -1:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.collect_trajectories()
            )

            timesteps_so_far += np.sum(batch_lens)

            iterations_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = timesteps_so_far
            self.logger["i_so_far"] = iterations_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            # Normalize advantages (mentioned in OpenAI implementation)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update actor network
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if iterations_so_far % self.save_freq == 0 and iterations_so_far != 0:
                self.save_model()

            self.wins_percent_queue.append(
                self.wins_queue.count(1) * 100 / len(self.wins_queue)
            )

            if (
                sum(self.wins_percent_queue) / len(self.wins_percent_queue)
                > self.break_after_x_continuous_win_percent and len(self.wins_percent_queue) >= self.how_many_consecutive_wins_to_break
            ):
                return

    def save_model(self, path=None):
        current_time = time.time()

        if path is None:
            path = f"./checkpoints/"

        path = f"{path}{self.current_agent_player}/{current_time}/"

        # make sure the directories exist
        os.makedirs(f"{path}", exist_ok=True)

        torch.save(
            self.actor.state_dict(),
            f"{path}actor.pth",
        )

        torch.save(
            self.critic.state_dict(),
            f"{path}critic.pth",
        )

    def collect_trajectories(self):
        """
        Collects a set of trajectories using the actor network

        Return:
                batch_obs - batch of episodes (will be the exact same as that returned by the hex environment)
                batch_acts - the actions collected this batch
                batch_log_probs - the log probabilities of each action taken this batch.
                batch_rtgs - the Rewards-To-Go of each timestep in this batch.
                length_of_each_episode_in_batch - the lengths of each episode this batch.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        length_of_each_episode_in_batch = []

        # Episodic data. Keeps track of rewards per episode, will¯ get cleared
        # upon each new episode
        rewards_per_episode = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        should_render = False
        if (
            self.render_every_x_iterations is not None
            and self.logger["i_so_far"] % self.render_every_x_iterations == 0
        ):
            should_render = True

        number_of_episodes_so_far = 0
        while number_of_episodes_so_far < self.episodes_per_batch:
            number_of_episodes_so_far += 1
            rewards_per_episode = []  # rewards collected per episode

            # Reset the environment
            self.env.reset()
            observation = self.env.observe(self.current_agent_player)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if should_render:
                    self.env.render()

                terminated = self.env.terminations[self.current_agent_player]
                truncated = self.env.truncations[self.current_agent_player]

                done = terminated | truncated

                if done:
                    # If we lose, we want to make the last step the negative reward
                    reward = self.env.rewards[self.current_agent_player]
                    if reward < -2:
                        rewards_per_episode[-1] = reward
                    break

                # If we are not playing as the agent, we want to play based on passed strategy. TODO make easier to implement custom opponents
                if self.env.agent_selection != self.current_agent_player:
                    action = self.opponent.get_action(observation)
                    self.env.step(action)
                    continue

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(observation)

                # Calculate action and make a step in the env.
                action, log_prob = self.get_action(observation, self.actor)
                self.env.step(action)
                observation = self.env.observe(self.current_agent_player)
                reward = self.env.rewards[self.current_agent_player]

                if reward == -1:
                    reward = reward * self.step_reward_multiplier

                # Track recent reward, action, and action log probability
                rewards_per_episode.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            should_render = False

            if self.opponent.reset:
                # Reset the opponent after each episode
                self.opponent.reset()

            if (
                self.env.rewards[self.current_agent_player]
                > self.env.rewards[self.other_agent_player]
            ):
                self.wins_queue.append(1)
            else:
                self.wins_queue.append(0)

            # Track episodic lengths and rewards
            length_of_each_episode_in_batch.append(ep_t)
            batch_rewards.append(rewards_per_episode)

        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_to_go = self.compute_rtgs(batch_rewards)

        self.logger["batch_rews"] = batch_rewards
        self.logger["batch_lens"] = length_of_each_episode_in_batch

        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rewards_to_go,
            length_of_each_episode_in_batch,
        )

    def compute_rtgs(self, batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, model):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
                obs - the observation at the current timestep

        Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        logits = model(
            torch.tensor(
                np.expand_dims(self.transform_observation(obs).numpy(), axis=0)
            )
        )

        action_probs = F.softmax(logits, dim=-1)

        action_mask = torch.tensor(
            self.env.generate_info(self.current_agent_player)["action_mask"]
        )

        action_probs = action_probs * action_mask

        sum_t = action_probs.sum()

        action_probs = action_probs / sum_t

        action_probs = torch.tensor(action_probs)

        # Sample an action
        action = torch.multinomial(action_probs, num_samples=1).item()

        # Calculate the log probability for that action
        log_prob = torch.log(action_probs[0, action])

        return action, log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of action)

        Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        batch_obs = [self.transform_observation(obs) for obs in batch_obs]
        batch_obs = torch.stack(
            batch_obs
        )  # Shape: (batch_size, channels, height, width)

        # Query critic network for values (V)
        V = self.critic(batch_obs).squeeze()

        # Query actor network for logits
        logits = self.actor(batch_obs)

        # Convert logits to action probabilities
        action_probs = F.softmax(logits, dim=-1)

        # Calculate log probabilities of the taken actions
        log_probs = torch.log(
            action_probs[torch.arange(len(batch_acts)), batch_acts.long()]
        )

        return V, log_probs

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
                None

        Return:
                None
        """
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]]
        )

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"PLAYER: {self.current_agent_player}", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(
            f"Training time elapsed in min {(time.time() - self.start_learn_time) / 60}",
            flush=True,
        )
        print(
            f"Wins % in last {self.max_num_of_episodes_to_calculate_win_percent} episodes = {self.wins_queue.count(1) * 100 / len(self.wins_queue)}%",
            flush=True,
        )
        
        if(len(self.wins_percent_queue) > 0):
            print(f"Wins % breakout percentage = {sum(self.wins_percent_queue) / len(self.wins_percent_queue)}%", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
