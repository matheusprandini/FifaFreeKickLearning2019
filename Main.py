import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
from DQN_RGB import DQN_RGB
from DQN import DQN
from FifaEnv import FifaEnv
from scipy.stats import wilcoxon
from DynamicMLP import MLP
import scipy.misc
from scipy.misc import imresize

# Initialize Global Parameters
DATA_DIR = "Models/"
NUM_ACTIONS = 4 # number of valid actions
MAX_ACTIONS = 6 # If execute MAX_ACTIONS, then it's considered a loop
GAMMA = 0.9 # decay rate of past observations
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
NUM_EPOCHS_OBSERVE = 200
NUM_EPOCHS_TRAIN = 5000
NUM_EPOCHS_TEST = 100
STEPS_TARGET_NETWORK = 1

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN
	
def train_dqn_free_kicks():

    game_env = FifaEnv()
    dqn = DQN_RGB(NUM_ACTIONS)
    #dqn = DQN(NUM_ACTIONS)
	
    dqn.save_model('target_network')
    dqn.update_target_network()
	
    num_goals = 0
    num_steps = 0
    epochs = []
    avg_goals = []
    epsilon = INITIAL_EPSILON
	
    print('----- STARTING DQN AGENT -----')

    for e in range(NUM_EPOCHS):

        history_actions = []
        game_over = False
        goal = 0
        loss = 0.0

        time.sleep(1.5)
		
        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
        bug = game_env.check_bug()
		
        if end_training_session or bug:
            game_env.hard_reset()
		
        while bug:
            bug = game_env.check_bug()
		
        # get first state
        #frames = collections.deque(maxlen=4)
		
        x_t = game_env.observe_state()
		
        #frames.append(x_t)
		
        #s_t = dqn.preprocess_images(np.array(list(frames)))
        s_t = dqn.preprocess_image(x_t)
	
        while not game_over:
			
			# Updates the previous state (previous state = current state)
            s_tm1 = s_t
			
            #### Get next action ####
			
            # if len(history_actions) > MAX_ACTIONS, there's a movement loop. So shoot the ball
            if len(history_actions) < MAX_ACTIONS:			
	            # Observation action (random)
                if e < NUM_EPOCHS_OBSERVE:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                # Random or the best current action based on q-value (dqn model)
                else:
		            # Random (exploration)
                    if np.random.rand() <= epsilon:
                        a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                    # Best action (exploitation)
                    else:
                        q = dqn.model.predict(s_t)[0]
                        a_t = np.argmax(q)

                history_actions.append(a_t)
            else:
                a_t = np.random.randint(low=2, high=NUM_ACTIONS, size=1)[0]

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)
            #frames.append(x_t)			
            #s_t = dqn.preprocess_images(np.array(list(frames)))
            s_t = dqn.preprocess_image(x_t)

		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1

            # store experience
            dqn.experience.append((s_tm1, a_t, r_t, s_t, game_over))

            if e >= NUM_EPOCHS_OBSERVE:
                # finished observing, now start training
                # get next batch
                num_steps += 1
                X, Y = dqn.get_next_batch(NUM_ACTIONS, GAMMA, BATCH_SIZE)
                #X, Y = dqn.get_next_batch_2(NUM_ACTIONS, GAMMA, BATCH_SIZE)
                loss += dqn.model.train_on_batch(X, Y)
        
            if num_steps == STEPS_TARGET_NETWORK and STEPS_TARGET_NETWORK != 1:
                num_steps = 0
                dqn.update_target_network()

        # reduce epsilon gradually
        if epsilon > FINAL_EPSILON and e >= NUM_EPOCHS_OBSERVE:
            #epsilon = 4 / ((e - NUM_EPOCHS_OBSERVE + 1) ** (1/2))
            epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON) / (NUM_EPOCHS_TRAIN / 1.5))
			
        #if e >= NUM_EPOCHS_OBSERVE:
        num_goals += goal
        epochs.append((e + 1))
        avg_goals.append(float(num_goals / (e + 1)))
        
        print("Epoch {:04d}/{:d} | Loss {:.5f} | Epsilon: {:.3f} | Total Goals: {:d} | Epoch Goal: {:d}"
            .format(e + 1, NUM_EPOCHS, loss, epsilon, num_goals, goal))
				
        if ((e + 1) % NUM_EPOCHS_OBSERVE == 0 and e >= NUM_EPOCHS_OBSERVE):
            dqn.model.save(os.path.join(DATA_DIR, "drl-network-fifa-final.h5"), overwrite=True)
        
    dqn.model.save(os.path.join(DATA_DIR, "drl-network-fifa-final.h5"), overwrite=True)
    np.save("epochs.npy",np.array(epochs))
    np.save("avg_goals.npy",np.array(avg_goals))
	
    for layer in dqn.model.layers:
        print(layer.get_weights())
	
	
def test_dqn_free_kicks():

    game_env = FifaEnv()
    dqn = DQN_RGB(NUM_ACTIONS)
    #dqn = DQN(NUM_ACTIONS)
    data = []
	
    dqn.load_model("drl-network-fifa-final")
	
    '''for layer in dqn.model.layers:
        print(layer.get_weights())'''

    num_goals = 0
	
    print('----- TESTING DQN AGENT -----')
	
    time.sleep(3) 

    for e in range(NUM_EPOCHS_TEST):

        history_actions = []
        game_over = False
        goal = 0

        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
		
        if end_training_session:
            game_env.hard_reset()
		
        time.sleep(2)
		
        # get first state
        #frames = collections.deque(maxlen=4)
		
        x_t = game_env.observe_state()
		
        #frames.append(x_t)
		
        #s_t = dqn.preprocess_images(np.array(list(frames)))
        s_t = dqn.preprocess_image(x_t)
		
        while not game_over:
			
			# Updates the previous state (previous state = current state)
            s_tm1 = s_t
			
            #### Get next action ####
			
            # if len(history_actions) > MAX_ACTIONS, there's a movement loop. So shoot the ball
            if len(history_actions) < MAX_ACTIONS:			
				
                # Random (exploration)
                if np.random.rand() <= 0.05:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                # Best action (exploitation)
                else:
                    q = dqn.model.predict(s_t)[0]
                    a_t = np.argmax(q)

                history_actions.append(a_t)
            else:
                a_t = np.random.randint(low=2, high=NUM_ACTIONS, size=1)[0]

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)
            #frames.append(x_t)			
            #s_t = dqn.preprocess_images(np.array(list(frames)))
            s_t = dqn.preprocess_image(x_t)
			
		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1
				
        time.sleep(2)
			
        num_goals += goal
        
        print("Epoch {:04d}/{:d} | Total Goals: {:d} | Epoch Goal: {:d}"
            .format(e + 1, NUM_EPOCHS_TEST, num_goals, goal))

    return float(num_goals / NUM_EPOCHS_TEST)

def calculate_avg_goals():

    avg_goals = np.load("avg_goals.npy")
    epochs = np.load("epochs.npy")
    epochs = epochs - NUM_EPOCHS_OBSERVE
    print(len(epochs))
    plt.plot(epochs[NUM_EPOCHS_OBSERVE:], avg_goals[NUM_EPOCHS_OBSERVE:], color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Avg Goals')
    plt.savefig('training_rmsprop_drl.png')
    

train_dqn_free_kicks()
test_dqn_free_kicks()
calculate_avg_goals()