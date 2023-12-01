This is a successful SB3 RL mode trained to steer the car in carla 
Input: pre-processed semantic camera input which had been 
run through cropping of the bottom of image and run through
a pre-trained CNN model up until a certain layer
observation_space = spaces.Box(low=0.0, high=1.0,shape=(7, 18, 8), dtype=np.float32)
So the RL model uses this as input in shape=(7, 18, 8) 