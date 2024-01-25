# to test a model 
from stable_baselines3 import PPO
from environment import CarEnv

#update here
models_dir = "C:\\SelfDrive\\models\\1705442732"

env = CarEnv()
env.reset()

#and update here
model_path = f"{models_dir}\\500000.zip"
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