import gymnasium as gym
import numpy as np
import sumo_rl
from stable_baselines3 import DQN

# 1. LOAD THE CUSTOM CLASSES

class WaitTimeObservation(sumo_rl.ObservationFunction):
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
            wait_times.append(max_wait / 100.0)
            
        observation = np.array(phase_id + min_green + density + queue + wait_times, dtype=np.float32)
        return observation

    def observation_space(self):
        num_features = self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes)
        return gym.spaces.Box(low=0.0, high=np.inf, shape=(num_features,), dtype=np.float32)

def exponential_priority_reward(traffic_signal):
    total_penalty = 0.0
    is_yellow = 'y' in traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id).lower()

    for lane in traffic_signal.lanes:
        vehicles = traffic_signal.sumo.lane.getLastStepVehicleIDs(lane)
        for veh_id in vehicles:
            v_class = traffic_signal.sumo.vehicle.getVehicleClass(veh_id)
            wait_time = traffic_signal.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
            wait_time = min(wait_time, 300) 
            exponential_wait = wait_time ** 1.5 
            
            if v_class == 'emergency':
                total_penalty -= wait_time ** 2
            else:
                total_penalty -= exponential_wait 
                
    if is_yellow:
        total_penalty = total_penalty / 100.0
        
    return total_penalty / 1000.0

# 2. RUN THE SAVED MODEL (INFERENCE)

print("Loading the trained AI brain...")
model = DQN.load("models/dqn_traffic_model")

env = gym.make('sumo-rl-v0',
               net_file='nets/intersection.net.xml',
               route_file='nets/routes.rou.xml,nets/ambulances.rou.xml',
               use_gui=True, 
               num_seconds=3600,
               yellow_time=5,
               min_green=10,
               delta_time=15,
               observation_class=WaitTimeObservation, 
               reward_fn=exponential_priority_reward)

obs, info = env.reset()
done = False

print(" SUMO GUI IS NOW OPEN ")
print("1. Click on the SUMO window.")
print("2. Change the 'Delay (ms)' slider at the top to roughly 100.")
print("3. Click back into this terminal.")

input("\nPress ENTER here in the terminal when you are ready to watch the AI...")

print("\nSimulation running...")
total_reward = 0
steps = 0

# Trackers for our live metrics
seen_vehicles = set()
mean_wait_times = []
total_stopped_cars = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # --- METRIC COLLECTION ---
    # 1. Log every unique vehicle that appears on the map
    current_vehicles = env.unwrapped.sumo.vehicle.getIDList()
    seen_vehicles.update(current_vehicles)
    
    # 2. Pull live queue and wait time stats directly from the AI's step info
    if 'system_mean_waiting_time' in info:
        mean_wait_times.append(info['system_mean_waiting_time'])
    if 'system_total_stopped' in info:
        total_stopped_cars.append(info['system_total_stopped'])
    
    total_reward += reward
    steps += 1
    done = terminated or truncated

# Calculate exact throughput: (Total unique vehicles seen) - (Vehicles still stuck on map)
vehicles_left_at_end = len(env.unwrapped.sumo.vehicle.getIDList())
throughput = len(seen_vehicles) - vehicles_left_at_end

env.close()

# 3. CALCULATE THE METRICS

print(" PERFORMANCE METRICS: ")
print(f"Total AI Decisions Made: {steps}")
print(f"Cumulative Reward Score: {total_reward:.2f}")
print(f"Total Vehicles Cleared (Throughput): {throughput} vehicles")

if mean_wait_times and total_stopped_cars:
    avg_stop_time = sum(mean_wait_times) / len(mean_wait_times)
    avg_stopped_cars = sum(total_stopped_cars) / len(total_stopped_cars)
    max_stopped_cars = max(total_stopped_cars)
    
    print(f"Average Stop Time per Vehicle: {avg_stop_time:.2f} seconds")
    print(f"Average Stopped Cars per Cycle: {avg_stopped_cars:.1f} cars")
    print(f"Maximum Gridlock (Worst-case queue): {max_stopped_cars} cars")
else:
    print("\nError: Could not extract advanced metrics.")