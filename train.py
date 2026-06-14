from agent import Agent
import gymnasium as gym

# env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

agent = Agent(env=env, max_buffer_size=200000, target_update_interval=10000)

agent.train(episodes=1200, batch_size=32, wm_batch_size=32, offline_training_epochs=100, imagination_steps=4)


