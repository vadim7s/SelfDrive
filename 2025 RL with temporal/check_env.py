#from gymnasium.utils import env_checker
from stable_baselines3.common.env_checker import check_env
from carenv import CarEnv

env = CarEnv()
check_env(env)
