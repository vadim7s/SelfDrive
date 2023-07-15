from stable_baselines3 import PPO
from carenv import CarEnv

models_dir = "models/1688990205"

env = CarEnv()
env.reset()

model_path = f"{models_dir}/1000000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        print(reward)