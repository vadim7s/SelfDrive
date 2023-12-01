from stable_baselines3 import PPO
from carenv_steer_only_cnn_test import CarEnv

models_dir = "models/1700985333"

env = CarEnv()
env.reset()

model_path = f"{models_dir}/2500000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        #print('reward from current step: ',reward)