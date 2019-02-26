import cv2
import numpy as np
import pytesseract as pt
import time
from PIL import Image
from Agent import Agent
from Screen import Screen

class FifaEnv():

    def __init__(self):
        self.screen = Screen()
        self.agent = Agent()
        self.score = 0
		
    def check_bug(self):
        new_screen = self.screen.GrabScreen()[40:850,0:1400]
        ball_location = new_screen[640:680,700:740]
		
        if np.mean(ball_location[:, :, 0]) < 60:
            return True
        else:
            return False
		
    def check_end_of_episode(self):
        new_screen = self.screen.GrabScreen()
        epoch_over_screen_1 = new_screen[630:660,400:475]
        epoch_over_screen_2 = new_screen[630:660,220:290]
        resized_screen_1 = cv2.resize(epoch_over_screen_1, (200, 80))
        resized_screen_2 = cv2.resize(epoch_over_screen_2, (200, 80))
        image_1 = Image.fromarray(resized_screen_1.astype('uint8'), 'RGB')
        image_2 = Image.fromarray(resized_screen_2.astype('uint8'), 'RGB')
        ocr_result_1 = pt.image_to_string(image_1)
        ocr_result_2 = pt.image_to_string(image_2)
		
        if ocr_result_1 == "REPETIR" or ocr_result_2 == "REPETIR":
            return True
        else:
            return False
			
    def execute_action(self, action):
	
        game_step_over = False
		
        if action == 0:
            #print('Action: Move to the left')
            self.agent.left()
        elif action == 1:
            #print('Action: Move to the right')
            self.agent.right()
        elif action == 2:
            #print('Action: Execute a low shoot')
            self.agent.low_shoot()
        else:
            #print('Action: Execute a high shoot')
            self.agent.high_shoot()
			
        if action in [2,3]:
            game_step_over = True
            time.sleep(3)
        else:
            time.sleep(1)
			
        return game_step_over
			
    def get_reward(self, screeshot, executed_action, game_step_over):
	
        time.sleep(1)

        reward_screen = screeshot[90:140,1270:1330]
        resized_screen = cv2.resize(reward_screen, (800, 300))
        image = Image.fromarray(resized_screen.astype('uint8'), 'RGB')
        ocr_result = pt.image_to_string(image)

        new_score = 0
        
        if ocr_result:
            ocr_reward = (''.join(c for c in ocr_result if c.isdigit() and c != ' ' and c != ''))
			
            if ocr_reward:
                new_score = int(ocr_reward)
            else:
                new_score = self.score
		
        reward = -0.05
		
		## Original
        if executed_action in [0,1]:
		    # Hasn't shooted the ball yet (Movement actions)
            reward = -0.05	
        elif np.abs(new_score - self.score) > 200:
		    # Scored a goal
            reward = 1
        elif game_step_over:
		    # Shooted the ball but didn't score
            reward = -1
			
        # Modified
        '''if executed_action in [0,1]:
		    # Hasn't shooted the ball yet (Movement actions)
            reward = -0.05	
        elif np.abs(new_score - self.score) > 1000:
		    # Scored a goal hitting a target
            reward = 1
        elif np.abs(new_score - self.score) > 200:
			# Scored a normal goal
            reward = 0.7
        elif np.abs(new_score - self.score) == 200:
			# Hit a post
            reward = 0.3
        elif game_step_over:
		    # Shooted the ball but didn't score
            reward = -1'''
		
        self.score = new_score
		
        return reward
		
    def observe_state(self):
         
        return self.screen.GrabScreen()
		
    def step(self, action):

	    # Execute an action
        game_step_over = self.execute_action(action)    	
        
        # Observe the new state
        new_state = self.observe_state()
		
        # Observe the reward
        reward = self.get_reward(new_state, action, game_step_over)
		
        return new_state[40:850,0:1400], reward, game_step_over
		
    def reset(self):
        self.score = 0
        self.agent.enter()
		
    def hard_reset(self):
        self.score = 0
        self.agent.escape()
        time.sleep(2)
        self.agent.enter()