from stable_baselines3.common import env_checker
from environment import CarEnv
env = CarEnv()
env_checker.check_env(env, warn=True, skip_render_check=True)