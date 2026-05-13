import gymnasium as gym
from gymnasium import spaces
import traci
import random
import traci.constants as tc
import numpy as np
import math
from .config import ENV_CONFIG, ACC_MAP, TRAIN_CONFIG
from collections import deque 
import time
import sys
from pathlib import Path

from plexe import Plexe, ACC, CACC, DRIVER, FAKED_CACC, SPEED, POS_X, POS_Y

VAR_SPEED = traci.constants.VAR_SPEED
VAR_POSITION = traci.constants.VAR_POSITION

class PlatoonEnv(gym.Env):

    def __init__(self, params=[], gui=False):
        self.action_space = spaces.Discrete(ENV_CONFIG['action_space_size'])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ENV_CONFIG['observation_space_size'],), dtype=np.float32)

        self.params = params
        self.gui = gui
        # self.total_steps = 0
        self.platoons = {}
        self.topology = {}
        self.collided_vehicles = []

        # episode stats
        self.failure = 0
        self.success = 0
        self.collision = 0
        self.event_steps = 0  # for the events of success join, failure or collision
        self.episode_steps = 0 # for reset sumo to train another flow demand

        self.event_reward = 0.0
        self.current_event = 0
        self.current_episode = 0

        self.jerk = 0.
        self.pre_act = 0.

        #BYIN: for speed track analysis
        self.track_egoID = 'vh_ego'
        self.track_fronterID = 'vh_fronter'
        self.count = 0
        self.lane_change= 0
        self.tracked = False

        #detectors
        self.det1 = ENV_CONFIG['det1']
        self.det2 = ENV_CONFIG['det2']
        self.det3 = ENV_CONFIG['det3']
        self.det4 = ENV_CONFIG['det4']
        
        self.det_off0 = ENV_CONFIG['off0_det']
        self.det_off1 = ENV_CONFIG['off1_det']
        
        self.det_off0_1 = ENV_CONFIG['off0_det_1']
        self.det_off1_1 = ENV_CONFIG['off1_det_1']
        
        #zones
        self.zone0 = ENV_CONFIG['zone0']
        self.zone1 = ENV_CONFIG['zone1']
        
        #Off_ramp
        self.off0 = 'OFF0'
        self.off1 = 'OFF1'
        
        self.platoon_lane = ENV_CONFIG['platoon_lane']
        self.mixed_lane = ENV_CONFIG['mixed_lane']

        self.ivd = ENV_CONFIG['ivd']
        self.warmup_duration = ENV_CONFIG['warmup_duration']

        self.mixed_lane_speed = ENV_CONFIG['mixed_lane_speed']
        self.platoon_lane_speed = ENV_CONFIG['platoon_speed']

        self.join_info = {"joiner": None,
                          "leader": None,
                          "fronter": None}

        self.joiner_active = False
        self.terminated = False
        self.truncated = False

        # dict of subscriptions
        self.sub = {}

        self.flow_values = ENV_CONFIG['flow_values']
        self.max_episode_steps = TRAIN_CONFIG["max_episode_steps"]

    def start(self):
        traci.start(self.params)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)

        print("Start warm up")
        self.warm_up()
        print("End warm up")

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _get_obs(self):

        '''
        Selects the joiner if not selected and returns the observation

        Returns:
            Observation(ndarray): a description of the state of the joiner
        '''

        if self.join_info["joiner"] is None:
            joiner , leader, platoon_fronter = self.select_joiner()
        else: 
            joiner, leader, platoon_fronter = self.join_info.values()

        joiner_lane = traci.vehicle.getLaneIndex(joiner)
        joiner_pos = traci.vehicle.getPosition(joiner)

        leaders = list(self.platoons.keys())
        index = leaders.index(leader)
        if len(leaders) > index + 1:
            platoon_follower = leaders[index+1]
        else:
            platoon_follower = None

        fronter, follower, plane_fronter, plane_follower = None, None, None, None    

        min_dist = {
            'fronter': ENV_CONFIG['max_caption_range'],
            'follower': ENV_CONFIG['max_caption_range'],

            'plane_fronter': ENV_CONFIG['max_caption_range'],
            'plane_follower': ENV_CONFIG['max_caption_range']
        }
        
        joiner_neighbors = traci.vehicle.getContextSubscriptionResults(joiner)
        veh_list = traci.vehicle.getIDList()
        for veh in joiner_neighbors:
            if veh in veh_list:
                veh_pos = joiner_neighbors[veh][VAR_POSITION]
                veh_lane = traci.vehicle.getLaneIndex(veh)
                veh_edge = traci.vehicle.getRoadID(veh)
                if joiner_lane == veh_lane and veh_edge != self.off0 and veh_edge != self.off1: 
                    if joiner_pos[0] < veh_pos[0] and (veh_pos[0] - joiner_pos[0]) <= min_dist['fronter']:
                        fronter = veh
                        min_dist['fronter'] = veh_pos[0] - joiner_pos[0]
                    elif joiner_pos[0] >= veh_pos[0] and (joiner_pos[0] - veh_pos[0])  <= min_dist['follower']:
                        follower = veh
                        min_dist['follower'] = joiner_pos[0] - veh_pos[0]
                if self.platoon_lane == veh_lane:
                    if joiner_pos[0] < veh_pos[0] and (veh_pos[0] - joiner_pos[0]) <= min_dist['plane_fronter']:
                        plane_fronter = veh
                        min_dist['plane_fronter'] = veh_pos[0] - joiner_pos[0]
                    elif joiner_pos[0] >= veh_pos[0] and (joiner_pos[0] - veh_pos[0])  <= min_dist['plane_follower']:
                        plane_follower = veh
                        min_dist['plane_follower'] = joiner_pos[0] - veh_pos[0]

        # Distances
        d_fronter_joiner = min_dist['fronter']
        d_joiner_follower = min_dist['follower']
        d_plane_fronter_joiner = min_dist['plane_fronter']
        d_joiner_plane_follower = min_dist['plane_follower']

        platoon_fronter_pos = traci.vehicle.getPosition(platoon_fronter)
        d_platoon_fronter_joiner = platoon_fronter_pos[0] - joiner_pos[0]


        if platoon_follower is not None:
            platoon_follower_pos = traci.vehicle.getPosition(platoon_follower)
            d_joiner_platoon_follower = joiner_pos[0] - platoon_follower_pos[0]  # BYIN: or we can use platoon_fronter_pos[0] - platoon_follower_pos[0] instead
        else:
            d_joiner_platoon_follower = ENV_CONFIG['max_caption_range']

        # Speeds 
        joiner_speed = traci.vehicle.getSpeed(joiner)

        if fronter is not None:
            fronter_speed = traci.vehicle.getSpeed(fronter)
        else:
            fronter_speed = self.mixed_lane_speed
            
        if follower is not None:
            follower_speed = traci.vehicle.getSpeed(follower)
        else:
            follower_speed = self.mixed_lane_speed

        if plane_fronter is not None:
            plane_fronter_speed = traci.vehicle.getSpeed(plane_fronter)
        else:
            plane_fronter_speed = self.platoon_lane_speed

        if plane_follower is not None:
            plane_follower_speed = traci.vehicle.getSpeed(plane_follower)
        else:
            plane_follower_speed = self.platoon_lane_speed

        platoon_fronter_speed = traci.vehicle.getSpeed(platoon_fronter)

        if platoon_follower is not None:
            platoon_follower_speed = traci.vehicle.getSpeed(platoon_follower) 
        else: 
            platoon_follower_speed = self.platoon_lane_speed


        observation = np.array([joiner_speed, fronter_speed, follower_speed, plane_fronter_speed, plane_follower_speed, platoon_fronter_speed, platoon_follower_speed,
                                d_fronter_joiner, d_joiner_follower, d_plane_fronter_joiner, d_joiner_plane_follower, d_platoon_fronter_joiner, d_joiner_platoon_follower,
                                joiner_lane, self.platoon_lane], dtype=np.float32)
        
        
        return observation


    def _get_info(self):
        return {"steps": self.event_steps,
                "reward": self.event_reward,
                "collisions" : self.collision,
                "successes" : self.success,
                "failures" : self.failure,
                }

    def _get_reward(self):
        '''
        Based on what Happened after applying the last action, return a reward

        Returns:
            reward (int): a value describing how good the last action was
        '''
        reward = 0
        joiner, _, fronter = self.join_info.values()

        # COLLISION
        if self.collision: 
            self.terminated = True
            reward += ENV_CONFIG['collision_penalty']
            print("collision")

        if joiner is not None and fronter is not None:
            joiner_pos = traci.vehicle.getPosition(joiner)
            fronter_pos = traci.vehicle.getPosition(fronter)
            
            # SUCCESSFUL JOIN
            if traci.vehicle.getLeader(joiner) and traci.vehicle.getLeader(joiner)[0] == fronter:
                self.terminated = True
                self.success += 1
                reward += ENV_CONFIG['succesful_join_reward']
                if self.gui:
                    traci.vehicle.setColor(self.join_info["joiner"], (0, 255, 0, 255))
                print("!!!!!!! SUCCESS !!!!!!!!!!!")
                                
                d = fronter_pos[0] - joiner_pos[0]

                #BYIN : results analysis
                if (ENV_CONFIG['test'] or ENV_CONFIG['test_baseline']) and ENV_CONFIG['save_dist_speed'] :
                    file_path = TRAIN_CONFIG['algo'] + '_join_distance.txt'
                    with open(file_path, "a") as f:
                        f.write(str(d) + '\n')

                if d < ENV_CONFIG['min_dist']:
                    print("distance too short")
                    reward -= ENV_CONFIG['short_dist_pen'] * (ENV_CONFIG['min_dist'] - d)
                if d > ENV_CONFIG['max_dist']:
                    print("distance too long")
                    print(d)
                    reward -= ENV_CONFIG['long_dist_pen'] * (d - ENV_CONFIG['max_dist'])

            # FAILED JOIN
            elif joiner in self.left_vehicles or fronter in self.left_vehicles or \
                traci.vehicle.getLaneIndex(joiner) == self.platoon_lane and (not traci.vehicle.getLeader(joiner) or \
                                                                  (traci.vehicle.getLeader(joiner) and traci.vehicle.getLeader(joiner)[0] != fronter)):
                self.terminated = True
                print("Failure ///////////////")
                if self.gui:
                    traci.vehicle.setColor(joiner, (255, 0, 0, 255))
                if joiner in self.left_vehicles or fronter in self.left_vehicles  : print("VEHICLE LEFT !")
                elif traci.vehicle.getLaneIndex(joiner) == self.platoon_lane:
                    print("JOIN IN WRONG POSITION")
                    # Delete vehicle if join in the wrong position 
                    self.delete_vehicle(joiner)
                reward += ENV_CONFIG['failure_penalty']
                self.failure = 1

            else: 
                # DELAY
                reward += ENV_CONFIG['delay_penalty']
                # BYIN: add comfort by vehicle jerk
                if abs(self.jerk ) > 4:
                    reward += ENV_CONFIG['comfort_penalty'] * abs(self.jerk)
                    print("VEHICLE JERK")
        return reward


    def reset_(self, seed = None, options= None):
        '''
        Resets the environement to start a new state

        Returns:
            observation: The initiale state
            info: Initiale information about the episode
        '''
        self.current_episode += 1
        self.event_steps = 0
        print(f"########## Reward = {self.event_reward} ############")
        print(f"############################")
        self.reset_join_info(success_join=self.success)
        self.event_reward = 0.0
        self.collision = 0
        self.failure = 0
        self.success = 0      
        self.terminated = False

        self.jerk = 0.
        self.pre_act = 0.

        # used to start traci the first time this function is called
        if not traci.isLoaded():
            self.start()
        
        observation = self._get_obs()
        info = self._get_info()

        print(f"########## Event {self.current_episode} ##########")
        print("################################")

        return observation, info

    def reset(self, seed=None, options=None):
        '''
        Resets the environement to start a new state

        Returns:
            observation: The initiale state
            info: Initiale information about the episode
        '''
        super().reset(seed=seed)
        self.current_event +=1
        # used to start traci the first time this function is called
        if not traci.isLoaded():
            self.current_episode = 1
            self.generate_flow_file()
            print(f"Flow_0 = {self.flow_0}")
            print(f"Flow_1 = {self.flow_1}")
            self.start()
        # reset sumo if max_episode_steps reaches
        elif self.truncated:
            self.reset_sumo()

        # reset basic variables after each event
        self.reset_variables()

        observation = self._get_obs()
        info = self._get_info()

        print(f"########## Episode {self.current_episode} and Event {self.current_event} ##########")
        print("################################")

        return observation, info

    def reset_variables(self):

        self.event_steps = 0

        print(f"########## Reward = {self.event_reward} ############")
        print(f"############################")

        self.reset_join_info(success_join=self.success)

        self.event_reward = 0.0
        self.collision = 0
        self.failure = 0
        self.success = 0

        self.terminated = False  #reset for current event

        self.jerk = 0.
        self.pre_act = 0.

    def reset_sumo(self):
        self.current_episode += 1
        self.current_event = 0
        self.episode_steps = 0

        self.platoons = {}
        self.topology = {}
        self.collided_vehicles = []
        self.join_info = {"joiner": None,
                          "leader": None,
                          "fronter": None}
        self.sub = {}

        if traci.isLoaded():
            traci.close()

        self.generate_flow_file()

        print(f"Flow_0 = {self.flow_0}")
        print(f"Flow_1 = {self.flow_1}")

        self.start()
        self.truncated = False #reset for current episode

    # BYIN: this is the decision step
    def step(self, action):
        # self.total_steps += 1
        self.event_steps += 1
        self.episode_steps += 1

        sim_steps_per_decision = 5 # 0.5 s
        lane_change_active_dur = 2 # seconds

        joiner, leader, fronter = self.join_info.values()

        if joiner is not None:
            self.joiner_active = True  # BYIN join

        if ENV_CONFIG['train']:
            traci.vehicle.setLaneChangeMode(joiner, 0) # BYIN totally disable SUMO control
        if ENV_CONFIG['test']:
            traci.vehicle.setLaneChangeMode(joiner, 0) # BYIN totally disable SUMO control
        if ENV_CONFIG['test_baseline']:
            traci.vehicle.setLaneChangeMode(joiner, 512) # BYIN hybrid control that RL combines a controlled, safe SUMO lane change

        if action == ENV_CONFIG['change_lane_action']:   # action = 0
            traci.vehicle.setVehicleClass(joiner, 'hov')
            # pos = traci.vehicle.getLanePosition(joiner)
            # lane = traci.vehicle.getLaneID(joiner)
            # lane = lane[:-1] +  str(self.platoon_lane)   # platoon lane ID

            #BYIN: attention: this should be added after simulationStep where the maneuver of lane change occurs
            # self.plexe.set_fixed_lane(joiner, self.platoon_lane, safe=False)
            # self.plexe.set_active_controller(joiner, CACC)
            # self.plexe.set_cc_desired_speed(joiner, self.platoon_lane_speed)
            # self.plexe.set_path_cacc_parameters(joiner, distance=self.ivd)
            #AKASMI:
            # if ENV_CONFIG['train']:
            #     traci.vehicle.moveTo(joiner, lane, pos)    # For training
            # if ENV_CONFIG['test']:
            #     traci.vehicle.changeLane(joiner, 1, 10)   # For testing
            traci.vehicle.changeLane(joiner, laneIndex=1,
                                     duration=lane_change_active_dur)  # BYIN : it is not the time to complete lane change but lane change command remains active
            traci.vehicle.updateBestLanes(joiner)

            self.pre_act = 0.
            # BYIN: attention: this should be added after simulationStep where the maneuver of lane change occurs
            # if joiner not in self.platoons[leader]["members"]:
            #     self.platoons[leader]["members"].append(joiner)
        if ENV_CONFIG['action_for_speed'] and action != ENV_CONFIG['change_lane_action']:  # BYIN: new added for other actions; DO NOT set Plexe controller
            traci.vehicle.setSpeedMode(joiner, 31) # BYIN: Mostly manual (no safety), when 0 --> Fully manual, everything disabled.
            a = ACC_MAP[action]
            traci.vehicle.slowDown(joiner, min(self.mixed_lane_speed, max(0, traci.vehicle.getSpeed(joiner) + a/2)), 0.5)
            self.jerk = (a - self.pre_act)/0.5
            self.pre_act = a

            # BYIN: track vh for speed analysis
            if not self.tracked:
                self.track_egoID = joiner
                # self.track_fronterID = fronter
                self.tracked = True
                self.lane_change = 0

        for _ in range(sim_steps_per_decision):
            stop = self.simulationStep()
            if stop:
                break
            # BYIN: update self.joiner_active (if it is deleted or not in simulationStep) and add joiner to platoon members
            if self.joiner_active:
                if traci.vehicle.getLaneIndex(joiner) == self.platoon_lane:
                    self.plexe.set_active_controller(joiner, CACC)
                    self.plexe.set_cc_desired_speed(joiner, self.platoon_lane_speed)
                    self.plexe.set_path_cacc_parameters(joiner, distance=self.ivd)
                    if leader in self.platoons and joiner not in self.platoons[leader]["members"]:
                        self.platoons[leader]["members"].append(joiner)
                    self.joiner_active = False

                    # BYIN: track vh for speed analysis
                    if self.track_egoID == joiner and self.lane_change == 0:
                        self.tracked = True
                        self.lane_change = 1
                    else:
                        self.tracked = False


        reward = self._get_reward()
        
        self.event_reward += reward

        if not self.terminated:
            observation = self._get_obs()
        else: 
            observation = np.full(ENV_CONFIG['observation_space_size'], -1, dtype=np.float32)
            
        info = self._get_info()

        self.truncated = self.episode_steps >= self.max_episode_steps
        return observation, reward, self.terminated, self.truncated, info
    
    def select_joiner(self):

        found = False

        while not found:
            # select joiner
            edges = list(traci.edge.getIDList())
            edges.remove(self.off0)
            edges.remove(self.off1)
            joiner_edge = random.choice(edges)
            edge_vehicles = traci.lane.getLastStepVehicleIDs(f"{joiner_edge}_{self.mixed_lane}")
            potential_joiners = [veh for veh  in edge_vehicles if traci.vehicle.getTypeID(veh) == 'cav']
            if not potential_joiners:
                self.simulationStep()
                continue
            joiner = random.choice(potential_joiners)

            # select platoon
            available_platoons = []
            joiner_x, joiner_y = traci.vehicle.getPosition(joiner)
            for leader in self.platoons:
                fronter = self.platoons[leader]["members"][-1]
                fronter_x, fronter_y = traci.vehicle.getPosition(fronter)

                if fronter_x <= joiner_x + ENV_CONFIG['upper_bound'] and fronter_x >= joiner_x + ENV_CONFIG['lower_bound']:
                    available_platoons.append(leader)

            if available_platoons:
                leader = random.choice(available_platoons)
                fronter = self.platoons[leader]["members"][-1]
                found = True
            
            else:
                self.simulationStep()

        self.topology[joiner] = {"front": fronter, "leader": leader}
        self.join_info = {"joiner":joiner, "leader":leader, "fronter":fronter}
        self.platoons[leader]['state'] = 1

        if self.gui:
            traci.vehicle.setColor(joiner, (255, 255, 0, 255))
            traci.gui.trackVehicle("View #0", joiner)
            traci.gui.setZoom("View #0", 2500)

        if joiner not in self.sub.keys():
            traci.vehicle.subscribeContext(joiner, traci.constants.CMD_GET_VEHICLE_VARIABLE, ENV_CONFIG['max_caption_range'],
                                                   [VAR_SPEED, VAR_POSITION])
            traci.vehicle.addSubscriptionFilterLeadFollow(lanes=[0, 1])
            self.sub[joiner] = traci.vehicle.getContextSubscriptionResults(joiner)
  
        return joiner, leader, fronter
    

    def handle_collision(self):
        for veh in self.collided_vehicles:
            self.delete_vehicle(veh)

    def handle_left_vehicles(self):
        for veh in self.left_vehicles:
            self.delete_vehicle(veh)
            
    def handle_off_ramp_exit(self):
        off_exit = []
        zone0 = traci.inductionloop.getLastStepVehicleIDs(self.det_off0)
        zone0 += traci.inductionloop.getLastStepVehicleIDs(self.det_off0_1)
        zone1 = traci.inductionloop.getLastStepVehicleIDs(self.det_off1)
        zone1 += traci.inductionloop.getLastStepVehicleIDs(self.det_off1_1)
        
        current_vehicles = traci.vehicle.getIDList()
        # Exiting vehicles
        for veh in zone0:
            if veh in current_vehicles and traci.vehicle.getRouteID(veh) == 'm0':
                off_exit.append(veh)
                
        for veh in zone1:
            if veh in current_vehicles and traci.vehicle.getRouteID(veh) == 'm1':
                off_exit.append(veh)
                
        # Splitting platoons
        for veh in off_exit:
            # case 1: veh is a leader
            if veh in list(self.platoons.keys()):
                platoon_members = self.platoons[veh]["members"]
                if len(platoon_members) >= 2:
                    new_leader = platoon_members[1]
                    self.platoons[new_leader] = {"members": [], "state": 1, "ini_size":2}
                    self.platoons[new_leader]["members"]= platoon_members[1:]
                    self.platoons.pop(veh)
                    self.topology.pop(new_leader)
                    self.plexe.set_active_controller(new_leader, ACC)
                    
                    for v in self.topology:
                        if self.topology[v]["leader"] == veh: 
                            self.topology[v]["leader"] = new_leader
                    
                    if self.join_info["leader"] == veh:
                        self.join_info["leader"] = new_leader   
                        if self.join_info["fronter"] == veh:
                            self.join_info["fronter"] = new_leader
                            
                else:
                    for v in list(self.topology.keys()):
                        if self.topology[v]["leader"] == veh: 
                            self.topology.pop(v)
                    self.platoons.pop(veh)
                    if self.join_info["leader"] == veh:
                        self.reset_join_info() 
                    
                    
            elif veh in list(self.topology.keys()): 
                # case 2: veh is a platoon member
                leader = self.topology[veh]["leader"]
                members = self.platoons[leader]["members"]
                
                # last member of the platoon
                index = members.index(veh)
                
                if veh != members[-1]:
                    self.topology[members[index+1]]["front"] = members[index-1]
                else:
                    if self.join_info["fronter"] == veh:
                        self.join_info["fronter"] = members[index-1]
                        
                self.platoons[leader]["members"].remove(veh)
                self.topology.pop(veh)
                
                
            traci.vehicle.setParameter(veh, "laneChangeModel.lcStrategic", "1.2")
            self.plexe.set_active_controller(veh, ACC)
            self.plexe.set_fixed_lane(veh, self.mixed_lane, safe=False)
            self.plexe.set_cc_desired_speed(veh, self.mixed_lane_speed)
            traci.vehicle.changeLane(veh, 0, 100)


    def reset_join_info(self, success_join=False):
        joiner = self.join_info["joiner"]
        if traci.isLoaded() and joiner is not None:
            if not success_join:
                # BYIN: do this only when two actions (lane change or keep lane), don't use RL setup for speed control
                if ENV_CONFIG['action_for_speed']==False:
                    self.plexe.set_active_controller(joiner, ACC)
                    traci.vehicle.setSpeedMode(joiner, 31) # BYIN: Full SUMO control (CACC / IDM active)
                if self.gui:
                    traci.vehicle.setColor(joiner, (0, 0, 255, 255))
                if joiner in self.topology:
                    self.topology.pop(joiner)  
                
            self.join_info = {"joiner": None,
                            "leader": None,
                            "fronter": None}
                
        if traci.isLoaded() and self.join_info["leader"] in traci.vehicle.getIDList():
            self.platoons[self.join_info["leader"]]["state"] = 0

        # BYIN: if joiner is reset, then it is not active anymore
        self.joiner_active = False
    def warm_up(self):
        for _ in range(self.warmup_duration):
            self.simulationStep()

    def configure_new_vehicles(self, min_platoon_size=2, max_platoon_size=2):
        # get list of input vehicles in the last step
        if len(traci.simulation.getDepartedIDList()) != 0:
            for veh in traci.simulation.getDepartedIDList():
                wrong_join = False
                # individual cavs, in normal lanes when input
                if traci.vehicle.getTypeID(veh) == "passenger":
                    lane = traci.vehicle.getLaneIndex(veh)
                    if self.gui:
                        traci.vehicle.setColor(veh, (128, 0, 128, 255))
                if traci.vehicle.getTypeID(veh) == "cav":
                    self.plexe.set_active_controller(veh, ACC)
                    traci.vehicle.setSpeedMode(veh, 0)
                    self.plexe.set_cc_desired_speed(veh, self.mixed_lane_speed)
                    if self.gui:
                        traci.vehicle.setColor(veh, (0, 0, 255, 255))
                # cavs in the lane destined
                if traci.vehicle.getTypeID(veh) == "cav2":
                    self.plexe.set_active_controller(veh, ACC)
                    self.plexe.set_fixed_lane(veh, self.platoon_lane, False)
                    traci.vehicle.setSpeedMode(veh, 0)
                    self.plexe.set_cc_desired_speed(veh, self.platoon_lane_speed)
                    # No car in front
                    if traci.vehicle.getLeader(veh) is None:
                        self.platoons[veh] = {"ini_size": random.randint(min_platoon_size, max_platoon_size),
                                        "members": [veh], "state":0 }
                        if self.gui:
                            traci.vehicle.setColor(veh, (255, 255, 255, 255))

                    # At least one car in front, vid_front
                    else:
                        vid_front = traci.vehicle.getLeader(veh)[0]
                        leader_id = 0
                        # The car in front is not yet in any platoon, that is to say, a single leader
                        if vid_front in self.platoons.keys():
                            leader_id = vid_front
                        # The car in front is already in a platoon
                        elif vid_front in self.topology.keys():
                            leader_id = self.topology[vid_front]["leader"]

                        # The leader is a veh that did a wrong join
                        else:
                            wrong_join = True
                            
                        # The platoon reached the defined length or a joiner is trying to join the platoon, begin new platoon
                        if leader_id != 0 and self.platoons[leader_id]["ini_size"] <= len(self.platoons[leader_id]["members"])or \
                            leader_id != 0 and self.platoons[leader_id]["state"] == 1 or wrong_join or \
                             leader_id != 0 and traci.vehicle.getRouteID(leader_id) ==  traci.vehicle.getRouteID(veh)  :
                            self.platoons[veh] = {"ini_size": random.randint(min_platoon_size, max_platoon_size),
                                            "members": [veh], "state": 0}
                            if self.gui:
                                traci.vehicle.setColor(veh, (255, 255, 255, 255))
                        # else, add current vehicle as a member
                        else:
                            self.plexe.set_active_controller(veh, CACC)
                            self.plexe.set_path_cacc_parameters(veh, distance=self.ivd)
                            self.platoons[leader_id]["members"].append(veh)
                            self.topology[veh] = {"front": vid_front, "leader": leader_id}
                            if self.gui:
                                traci.vehicle.setColor(veh, (128, 128, 128, 255))

    def simulationStep(self):
        stop = False
        self.communicate()
        traci.simulationStep()

        #BYIN: track ego speed
        if (ENV_CONFIG['test'] or ENV_CONFIG['test_baseline']) and ENV_CONFIG['save_dist_speed']:
            file_name = TRAIN_CONFIG['algo'] + '_speed.txt'
            veh_ids = set(traci.vehicle.getIDList())
            #if self.tracked and {self.track_egoID, self.track_fronterID}.issubset(veh_ids):
            if self.tracked and self.track_egoID in veh_ids:
                self.count += 1
                ego_speed = traci.vehicle.getSpeed(self.track_egoID)
                # fronter_speed = traci.vehicle.getSpeed(self.track_fronterID)
                if self.count < 1000000:
                    with open(file_name, "a") as f:
                        # f.write(str(self.track_egoID) + ';' + str(ego_speed) + ';' + str(self.lane_change) + ';'+ str(self.track_fronterID) + ';' + str(fronter_speed) + '\n')
                        f.write(str(self.track_egoID) + ';' + str(ego_speed) + ';' + str(self.lane_change) + ';' + str(self.pre_act) + '\n')

        joiner, leader, fronter = self.join_info.values()
        self.configure_new_vehicles()
        
        self.collided_vehicles = traci.simulation.getCollidingVehiclesIDList()
        self.left_vehicles = traci.inductionloop.getLastStepVehicleIDs(self.det1) + traci.inductionloop.getLastStepVehicleIDs(self.det2)\
           + traci.inductionloop.getLastStepVehicleIDs(self.det3) + traci.inductionloop.getLastStepVehicleIDs(self.det4)
           
        if self.collided_vehicles:
            if self.gui:
                traci.gui.trackVehicle("View #0", self.collided_vehicles[0])
                traci.gui.setZoom("View #0", 2500)
                time.sleep(5)

            if self.join_info["joiner"] in self.collided_vehicles:
                self.collision = 1

            self.handle_collision()
            stop = True   

        self.handle_left_vehicles()
        #self.handle_off_ramp_exit()

        return stop
    
    def is_vehicle_known(self, vehicle_id):
        try:
            traci.vehicle.getPosition(vehicle_id)
            return True
        except traci.exceptions.TraCIException:
            return False

    def communicate(self):
        """
        Performs data transfer between vehicles, i.e., fetching data from
        leading and front vehicles to feed the CACC algorithm
        """
        try:
            for vid, l in self.topology.items():
                if "leader" in l.keys():
                    # get data about platoon leader
                    ld = self.plexe.get_vehicle_data(l["leader"])
                    # pass leader vehicle data to CACC
                    self.plexe.set_leader_vehicle_data(vid, ld)
                    # pass data to the fake CACC as well, in case it's needed
                    # self.plexe.set_leader_vehicle_fake_data(vid, ld)
                if "front" in l.keys():
                    # get data about platoon front
                    fd = self.plexe.get_vehicle_data(l["front"])
                    # pass front vehicle data to CACC
                    self.plexe.set_front_vehicle_data(vid, fd)
                    # compute GPS distance and pass it to the fake CACC
                    # distance = self.get_distance(vid, l["front"])
                    # self.plexe.set_front_vehicle_fake_data(vid, fd, distance)
        except traci.exceptions.FatalTraCIError as e:
            print(f"Fatal TraCI error during simulation step: {e}")
            return 

    def delete_vehicle(self, veh):
        # The veh is a platoon leader
        if veh in list(self.platoons.keys()): 
                platoon_members = self.platoons[veh]["members"]
                if len(platoon_members) >= 2:
                    new_leader = platoon_members[1]

                    self.platoons[new_leader] = {"members": [], "state": 1, "ini_size":2}
                    self.platoons[new_leader]["members"]= platoon_members[1:]
                    if self.gui:
                        traci.vehicle.setColor(new_leader, (255, 255, 255, 255))
                    self.platoons.pop(veh)
                    self.topology.pop(new_leader)
                    self.plexe.set_active_controller(new_leader, ACC)

                    for v in self.topology:
                        if self.topology[v]["leader"] == veh:
                            self.topology[v]["leader"] = new_leader

                    for v in self.topology:
                        if self.topology[v]["front"] == veh:
                            self.topology[v]["front"] = new_leader

                    if self.join_info["leader"] == veh:
                        self.join_info["leader"] = new_leader
                        if self.join_info["fronter"] == veh:
                            self.join_info["fronter"] = new_leader

                else:
                    for v in list(self.topology.keys()):
                        if self.topology[v]["leader"] == veh: 
                            self.topology.pop(v)
                    self.platoons.pop(veh)
                    if self.join_info["leader"] == veh:
                        self.reset_join_info() 

        # The veh is a platoon member
        elif veh in list(self.topology.keys()): 
            leader = self.topology[veh]["leader"]
            members = self.platoons[leader]["members"]
            if veh in members:
                index = members.index(veh)
                
                if veh != members[-1]:
                    self.topology[members[index+1]]["front"] = members[index-1]
                else:
                    for v in list(self.topology.keys()):
                        if self.topology[v]["front"] == veh: 
                            self.topology[v]["front"] = members[index-1]

                if self.join_info["fronter"] == veh:
                    self.join_info["fronter"] = members[index-1]

                self.platoons[leader]["members"].remove(veh)

            self.topology.pop(veh)

        if self.join_info["joiner"] is not None and self.join_info["joiner"] == veh:
            self.reset_join_info()

        if veh in traci.vehicle.getIDList():
            traci.vehicle.remove(veh)
            if veh in list(self.sub.keys()):
                traci.vehicle.unsubscribeContext(veh, traci.constants.CMD_GET_VEHICLE_VARIABLE, ENV_CONFIG['max_caption_range'])
                self.sub.pop(veh, None)


    def get_veh_dist(self, veh1_pos, veh2_pos):
        if veh1_pos == -1 or veh2_pos == -1:
            return ENV_CONFIG['max_caption_range']
        else:
            return abs(veh1_pos - veh2_pos)



    def generate_flow_file(self):

        self.flow_0 = random.choice(self.flow_values)
        self.flow_1 = random.choice(self.flow_values)
        output_path = 'config/flows_episode.add.xml'

        xml = f"""
    <additional>

        <flow id="lane0"
              begin="0"
              end="1000000"
              departPos="base"
              departSpeed="max"
              departLane="0"
              vehsPerHour="{self.flow_0}"
              type="vmix"
              route="m" />

        <flow id="lane1"
              begin="0"
              end="1000000"
              departPos="base"
              departSpeed="max"
              departLane="1"
              vehsPerHour="{self.flow_1}"
              type="cav2"
              route="m" />

    </additional>
    """

        with open(output_path, "w") as f:
            f.write(xml)

