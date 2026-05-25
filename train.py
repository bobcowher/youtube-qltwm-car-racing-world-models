from agent import Agent
import gymnasium as gym

# env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

agent = Agent(env=env, max_buffer_size=500000)

# agent.train(episodes=1200)


