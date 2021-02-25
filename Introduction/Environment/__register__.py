import gym
import reward_matrix
from gym.envs.registration import register

R,prp = reward_matrix.heading(30)
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'basinGrid-v0' in env:
          print('Removed {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]

print("Renewal of Environment id : {} has done".format("basinGrid-v0"))
register(id ='basinGrid-v0',
    entry_point='Environment.Basin:Surrounding',
    max_episode_steps=2500,
    kwargs={'R' : R, 'prp' : prp,},
    )

