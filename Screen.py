import cv2
import numpy as np
import pytesseract as pt
import win32gui, win32ui, win32con, win32api
import time
import sys
import os
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0]))

class Screen():
    
    def __init__(self, region=None):
        self.region = region      
    
    def GrabScreen(self):
        # Done by Frannecklp
    
        hwin = win32gui.GetDesktopWindow()
    
        if self.region:
                left,top,x2,y2 = self.region
                width = x2 - left + 1
                height = y2 - top + 1
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)
    
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
    
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if __name__ == '__main__':
    screen = Screen()
    last_time = time.time()
    while (True):
        reward_screen = screen.GrabScreen()[630:660,220:290]
        resized_screen = cv2.resize(reward_screen, (200, 80))
        cv2.imshow('window', resized_screen)
        image = Image.fromarray(resized_screen.astype('uint8'), 'RGB')
        ocr_result = pt.image_to_string(image)
        print(ocr_result)
        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        