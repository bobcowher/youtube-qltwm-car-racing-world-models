import os
from re import T
import gymnasium as gym
import cv2
import torch
from torch._C import device
from torch.cuda import device_count
import torch.nn.functional as F
import random

from torch.nn.utils import stateless
from buffer import ReplayBuffer
from models import world_model
from models.q_model import QModel
from models.world_model import WorldModel
import datetime
from torch.utils.tensorboard.writer import SummaryWriter

class MixedSampler:

    def __init__(self, agent, real_ratio=0.5):
        self.agent = agent
        self.real_ratio = real_ratio

    def sample(self, batch_size, horizon):
        if random.random() < self.real_ratio:
            return self._sample_real(batch_size, horizon)
        
        return self._sample_imagined(batch_size, horizon)

    def _sample_real(self, batch_size, horizon):

        obs, actions, rewards, next_obs, dones = self.agent.memory.sample_buffer(batch_size * horizon)

        with torch.no_grad():
            states = self.agent.world_model.encode(self.agent.normalize_observation(obs)).squeeze(1)
            next_states = self.agent.world_model.encode(self.agent.normalize_observation(next_obs)).squeeze(1)
        rewards = rewards.float()
        return states, actions, rewards, next_states, dones

    def _sample_imagined(self,  batch_size, horizon):
        return self.agent.imagine_trajectory(batch_size, horizon)


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 20000,
                       target_update_interval: int = 10000) -> None:
        self.env = env
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.98
        
        self.imagine_epsilon = 1
        self.imagine_min_epsilon = 0.1
        self.imagine_epsilon_decay = 0.99


        self.target_update_interval = target_update_interval
        self.total_steps = 0

        self.gamma = 0.99

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        obs, _ = self.env.reset()
        obs = self.process_observation(obs)

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device
        )

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024, n_actions=self.env.action_space.n).to(self.device) 

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)

        print(f"Initializing agent on device {self.device}")

        self.q_model = QModel(
            action_dim=self.env.action_space.n,
            input_shape=obs.shape,
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=self.env.action_space.n,
            input_shape=obs.shape,
        ).to(self.device)

        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.0001)

    def normalize_observation(self, obs):
        return obs / 255.0
        

    def imagine_trajectory(self, batch_size, horizon):
        obs, _, _, _, _ = self.memory.sample_buffer(batch_size)
        obs = self.normalize_observation(obs)
        
        with torch.no_grad():
            embeds = self.world_model.encode(obs).squeeze(1) # (batch_size, embed_dim)

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        current_embeds = embeds

        for _ in range(horizon):

            with torch.no_grad():
                q_vals = self.q_model(current_embeds)
                best_actions = q_vals.argmax(dim=1)
                random_actions = torch.randint(0, self.env.action_space.n, (batch_size,), device=self.device)
                exploration_mask = (torch.rand(batch_size, device=self.device) < self.imagine_epsilon).long()
                action_idx = exploration_mask * random_actions + (1 - exploration_mask) * best_actions

                action_onehot = F.one_hot(action_idx, num_classes=self.env.action_space.n).float()

                next_embeds, rewards, dones = self.world_model.imagine_step(current_embeds, action_onehot)

                all_states.append(current_embeds)
                all_actions.append(action_idx)
                all_rewards.append(rewards.squeeze(-1))
                all_next_states.append(next_embeds)
                all_dones.append((dones.squeeze(-1) > 0.5).float())

                current_embeds = next_embeds

        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        next_states = torch.cat(all_next_states, dim=0)
        dones = torch.cat(all_dones, dim=0)

        return states, actions, rewards, next_states, dones



    def process_observation(self, obs):
        obs = cv2.resize(obs, (96,96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def select_action(self, obs, eval=False):
        if random.random() < self.epsilon and not eval:
            return random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.20, 0.20, 0.50, 0.05])[0]
        
        with torch.no_grad():
            obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
            return self.q_model(obs_t).argmax(dim=1).item()

    def train_world_model(self, batch_size):

        obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size)

        loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, next_obs, dones)

        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.world_model_optimizer.step()

        return (
            loss_dict['total'],
            loss_dict['recon'],
            loss_dict['dynamics'],
            loss_dict['reward'],
            loss_dict['done']
        )

    
    def train_step(self, batch_size):

        obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size)

        obs_norm = obs / 255.0
        next_obs_norm = next_obs / 255.0

        actions = actions.unsqueeze(1).long()
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()
        
        q_values = self.q_model(obs_norm)
        q_sa = q_values.gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_obs_norm).argmax(dim=1, keepdim=True)
            next_q = self.target_q_model(next_obs_norm).gather(1, next_actions)
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_sa, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.total_steps += 1

        return loss.item()

    def save(self):
        self.q_model.save_the_model("q_model", verbose=True)
        self.world_model.save_the_model("world_model", verbose=True)

    def save_best(self):
        self.q_model.save_the_model("q_model_best", verbose=True)
        self.world_model.save_the_model("world_model_best", verbose=True)

    def load(self):
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_the_model("q_model", device=self.device)
    
    def test(self, episodes=1):
        self.q_model.eval()

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)

            done = False
            episode_reward = 0.0
            episode_steps = 0 

            while not done:

                action = self.select_action(obs, eval=True)

                next_obs, reward, term, trunc, _ = self.env.step(action) 
                next_obs = self.process_observation(next_obs)
                done = term or trunc

                episode_reward += float(reward)
                episode_steps += 1

                obs = next_obs

            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")



    def train(self, episodes=1, batch_size=32, offline_training_epochs=1, wm_batch_size=1):
        run_tag = f'initial'
        writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'
        writer = SummaryWriter(writer_name)

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0 

            while not done:

                action = self.select_action(obs)

                next_obs, reward, term, trunc, _ = self.env.step(action) 
                next_obs = self.process_observation(next_obs)
                done = term or trunc

                self.memory.store_transition(obs, action, reward, next_obs, done)

                episode_reward += float(reward)
                episode_steps += 1

                if self.memory.can_sample(batch_size):
                    episode_loss += self.train_step(batch_size)

                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0

            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            total_combined_loss = 0.0
            total_reward_loss = 0.0
            total_done_loss = 0.0
            total_recon_loss = 0.0
            total_dynamics_loss = 0.0
            total_q_loss = 0.0
            total_imagine_reward = 0.0
            wm_updates = 0
            q_updates = 0

            for _ in range(offline_training_epochs):
                for _ in range(5):
                    combined_loss, reward_loss, done_loss, recon_loss, dynamics_loss = self.train_world_model(batch_size=wm_batch_size)
                    total_combined_loss += combined_loss
                    total_reward_loss += reward_loss
                    total_done_loss += done_loss
                    total_recon_loss += recon_loss
                    total_dynamics_loss += dynamics_loss
                    wm_updates += 1

            if(wm_updates > 0):
                avg_combined_loss = total_combined_loss / wm_updates
                avg_reward_loss = total_reward_loss / wm_updates
                avg_done_loss = total_done_loss / wm_updates
                avg_recon_loss = total_recon_loss / wm_updates
                avg_dynamics_loss = total_dynamics_loss / wm_updates

                writer.add_scalar("World Model/combined_loss", avg_combined_loss, episode)
                writer.add_scalar("World Model/reconstruction_loss", avg_recon_loss, episode)
                writer.add_scalar("World Model/dynamics_loss", avg_dynamics_loss, episode)
                writer.add_scalar("World Model/reward_loss", avg_reward_loss, episode)
                writer.add_scalar("World Model/done_loss", avg_done_loss, episode)



            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon", self.epsilon, episode)
            writer.add_scalar("Train/avg_q_loss", avg_loss, episode)

            if episode % 10 == 0:
                self.save()

