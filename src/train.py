import gymnasium as gym
import numpy as np
import sumo_rl
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. CUSTOM OBSERVATION SPACE (The AI's Eyes)

class WaitTimeObservation(sumo_rl.ObservationFunction):
    """
    Custom Observation Function that explicitly feeds the queue 
    waiting times into the neural network.
    """
    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self):
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]        
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        
        wait_times = []
        for lane in self.ts.lanes:
            veh_list = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            max_wait = 0.0
            for veh in veh_list:
                wait = self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if wait > max_wait:
                    max_wait = wait
            # Normalize the wait time (divide by 100) so the neural network can digest it safely
            wait_times.append(max_wait / 100.0)
            
        # Combine all data into one giant array
        observation = np.array(phase_id + min_green + density + queue + wait_times, dtype=np.float32)
        return observation

    def observation_space(self):
        num_features = self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes)
        return gym.spaces.Box(low=0.0, high=np.inf, shape=(num_features,), dtype=np.float32)

# 2. EXPONENTIAL REWARD FUNCTION

def exponential_priority_reward(traffic_signal):
    total_penalty = 0.0
    
    is_yellow = 'y' in traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id).lower()

    for lane in traffic_signal.lanes:
        vehicles = traffic_signal.sumo.lane.getLastStepVehicleIDs(lane)
        for veh_id in vehicles:
            v_class = traffic_signal.sumo.vehicle.getVehicleClass(veh_id)
            wait_time = traffic_signal.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
            
            wait_time = min(wait_time, 300) 
            
            # Smoother exponential curve
            exponential_wait = wait_time ** 1.5 
            
            if v_class == 'emergency':
                total_penalty -= wait_time ** 2
            else:
                total_penalty -= exponential_wait 
                
    # THE FIX: THE YELLOW FORGIVENESS
    # If the light is yellow, divide the total penalty by 100!
    # This removes the AI's fear of switching lights.
    if is_yellow:
        total_penalty = total_penalty / 100.0
        
    # Scale the final reward so the neural network remains stable
    return total_penalty / 1000.0

# 3. TRAINING PIPELINE
print("PHASE 1: FAST TRAINING (NO GUI)")
train_env = gym.make('sumo-rl-v0',
               net_file='nets/intersection.net.xml',
               route_file='nets/routes.rou.xml,nets/ambulances.rou.xml',
               out_csv_name='models/dqn_results',
               use_gui=False, 
               num_seconds=3600,
               yellow_time=5,
               min_green=10,
               delta_time=15,
               observation_class=WaitTimeObservation, # <--- INJECTING OUR CUSTOM EYES HERE
               reward_fn=exponential_priority_reward)

train_env = DummyVecEnv([lambda: train_env])

# We delete the old, confused model so it starts fresh!
# THE FIX: HYPERPARAMETER TUNING
model = DQN("MlpPolicy", 
            train_env, 
            learning_rate=5e-4,           # Slightly slower, more stable learning
            learning_starts=2000,         # Gather more random data before guessing
            train_freq=1, 
            target_update_interval=500, 
            exploration_fraction=0.5,     # FORCE IT TO EXPLORE 50% OF THE TIME!
            exploration_final_eps=0.05, 
            verbose=1)

print("Training in background...")
model.learn(total_timesteps=50000)        # Give it enough time to learn (50,000 steps)
model.save("models/dqn_traffic_model")
train_env.close()

# 4. TESTING PIPELINE

print("PHASE 2: WATCHING THE AI (WITH GUI)")
test_env = gym.make('sumo-rl-v0',
               net_file='nets/intersection.net.xml',
               route_file='nets/routes.rou.xml,nets/ambulances.rou.xml',
               use_gui=True, 
               num_seconds=3600,
               yellow_time=5,
               min_green=10,
               delta_time=15,
               observation_class=WaitTimeObservation,
               reward_fn=exponential_priority_reward)

obs, info = test_env.reset()
done = False

print("Starting simulation! Make sure to adjust the DELAY slider in SUMO!")
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

test_env.close()
print("Simulation complete!")