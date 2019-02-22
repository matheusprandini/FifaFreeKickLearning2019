import time
from Keys import Keys, A, D, LCTRL, ESC, ENTER, SPACE

class Agent():

    def __init__(self):
	    self.keyboard = Keys()

    def right(self):
        self.keyboard.PressKey(D)
        time.sleep(0.2)
        self.keyboard.ReleaseKey(D)

    def left(self):
        self.keyboard.PressKey(A)
        time.sleep(0.2)
        self.keyboard.ReleaseKey(A)
    
    def high_shoot(self):
        self.keyboard.PressKey(SPACE)
        time.sleep(0.22)
        self.keyboard.ReleaseKey(SPACE)
    
    def low_shoot(self):
        self.keyboard.PressKey(LCTRL)
        self.keyboard.PressKey(SPACE)
        time.sleep(0.25)
        self.keyboard.ReleaseKey(SPACE)
        time.sleep(0.2)
        self.keyboard.ReleaseKey(LCTRL)
		
    def escape(self):
        self.keyboard.PressKey(ESC)
        time.sleep(0.5)
        self.keyboard.ReleaseKey(ESC)
		
    def enter(self):
        self.keyboard.PressKey(ENTER)
        time.sleep(0.5)
        self.keyboard.ReleaseKey(ENTER)