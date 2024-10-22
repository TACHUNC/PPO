
from pcgym import make_env
import numpy as np
from stable_baselines3 import PPO

nsteps = 150
T = 3
SP = {
    'X1': [0 for i in range(int(nsteps))],
    #'X2': [0 for i in range(int(nsteps))] 
}

#Continuous box action space
action_space = {
    'low': np.array([-1]),
    'high':np.array([1]) 
}
#Continuous box observation space
observation_space = {
    'low' : np.array([-1,-1,-1]),
    'high' : np.array([1,1,1])  
}

r_scale ={
    'X1': 100,
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, #Setpoint
    'o_space' : observation_space, #Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([1,-1,0.]), # Initial conditions (torch.tensor)
    'model': 'nonsmooth_control_ode', #Select the model
    'r_scale': r_scale, #Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': False, #Normalise the actions
    'normalise_o':False, #Normalise the states,
    'noise':False, #Add noise to the states
    'integration_method': 'casadi', #Select the integration method
    'noise_percentage':0 #Noise percentage
}
env = make_env(env_params)
# Load the saved policy
bang_pol = PPO('MlpPolicy', env, verbose=1,learning_rate=1e-3)
bang_pol.learn(total_timesteps=1e4)
bang_pol.save('pse_track_ppo.zip')
bang_pol.load('pse_track_ppo.zip')

# Evaluate the policy and plot the rollout
evaluator, data = env.plot_rollout({'SAC': bang_pol}, reps=1, oracle=True, dist_reward=True, MPC_params={'N': 10, 'R': 0})