from sympy import false

CONFIG = "acc_actions"
# """CHANGE HYPER PARAMETERS HERE""" ###################################################################################
TRAIN_CONFIG = {
    'gpu': '0',                                 # GPU # 0 is gpu, 1 is cpu
    'n_env': 1,                                 # Multi-processing environments
    'lr': 2e-04,                                # Learning rate
    'gamma': 0.9,                               # Discount factor
    'eps_start': 1,                             # Epsilon start
    'eps_min': 0.01,                            # Epsilon min
    'eps_dec': 2e5,                             # Epsilon decay rate
    'eps_dec_exp': True,                        # Epsilon exponential decay
    'bs': 32,                                   # Batch size 32
    'min_mem': 5000,                            # Replay memory buffer min size 
    'max_mem': 10000,                           # Replay memory buffer max size  
    'target_update_freq': 5000,                 # Target network update frequency
    'target_soft_update': True,                 # Target network soft update
    'target_soft_update_tau': 1e-03,            # Target network soft update tau rate
    'save_freq': 1000,                           # Save frequency 1000: straining steps
    'log_freq': 200,                            # Log frequency 5000: straining steps
    'save_dir': './save/' + CONFIG + "/",       # Save directory
    'log_dir_train': './logs/train/' + CONFIG + "/", # Log directory (uncomment for training)
    'log_dir_test': './logs/test/' + CONFIG + "/",   # Log directory (uncomment for testing)                             
    'load': True,                               # Load model 
    'repeat': 0,                                # Repeat action
    'max_episode_steps': 1000,                  # Time limit episode (decision) steps 1000 (0.1s/step)
    'max_total_steps': 1000000,                  # Max total training steps e.g.,1000000 if > 0, else (if =0) inf training
    'max_total_episodes_test': 100,           # Max total testing episodes, default 10000
    'algo': 'DQNAgent'                    # DQNAgent
                                                # DoubleDQNAgent
                                                # DuelingDoubleDQNAgent
                                                # PerDuelingDoubleDQNAgent
                                                # PPORLAgent
}

ENV_CONFIG = {
    # choice one of them regarding lane change with a safe mechanism (test_baseline) or not (train, test)
    'train': True,
    'test': False,
    'test_baseline': False,

    # Actions executed for speed adjustement
    'action_for_speed': True,

    # Platoon selection distance Interval (comparing the joiner position to the position of the last veh in the platoon)
    'lower_bound' : -150,
    'upper_bound' : 5,

    # inter-vehicle distance
    'ivd' : 5,
    
    'observation_space_size' : 15,
    'action_space_size' : 6, # or 2 (keep lane or lane change)
    'change_lane_action' : 0,

    'warmup_duration' : 1200,

    # rewards
    'collision_penalty' : - 10000,
    'succesful_join_reward' : 100,
    'delay_penalty' : -1,
    'comfort_penalty' : -1,
    'failure_penalty' : -50,

    #lanes
    "mixed_lane": 0,
    "platoon_lane": 1,

    #speeds
    'mixed_lane_speed' : 100 / 3.6, #100 km/h
    'platoon_speed' : 120 / 3.6,     #120 km/h

    # detectors names
    'det1' : 'D1',
    'det2' : 'D2',
    'det3' : 'D3',
    'det4' : 'D4',
    
    'off0_det' : 'det_OFF0', 
    'off1_det' : 'det_OFF1',
    'off0_det_1' : 'det_OFF0.1', 
    'off1_det_1' : 'det_OFF1.1',
    
    #zones
    'zone0': 'E23',
    'zone1': 'E45',
    

    'max_caption_range': 200,  #200 m

    # distance interval between the joiner and last member of the target platoon
    'min_dist' : 10 ,  # 10 meters 
    'max_dist' : 20,   # 20 meters

    # Penality factors (case of distance too long or too short) 
    'long_dist_pen': 0.5,
    'short_dist_pen': 2,

}

ACC_MAP = {
    1: 2.0,
    2: 1.0,
    3: 0.0,
    4: -1.0,
    5: -2.0
}


