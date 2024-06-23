import pygame
from abc import ABC, abstractmethod

"""
Source Video
https://www.youtube.com/watch?v=XmSv2V69Y7A
"""

vec = pygame.math.Vector2

class Camera:
    def __init__(self, aicar):
        self.aicar = aicar
        self.offset = vec(0,0)
        self.offset_float = vec(0,0)
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1600, 1200
        self.CONST = vec(0,0) #Need to change

    def set_method(self, method):
        self.method = method
        
    def up_scroll(self):
        self.method.scroll()
    

class CamScroll(ABC):
    def __init__(self, camera, aicar):
        self.camera = camera
        self.aicar = aicar

    @abstractmethod
    def scroll(self):
        pass

class Follow(CamScroll):
    def __init__(self, camera, aicar):
        CamScroll.__init__(self, camera, aicar)
    
    def scroll(self):
        self.camera.offset_float.x += (self.aicar.rect.x - self.camera.offset_float.x + self.camera.CONST.x)
        self.camera.offset_float.y += (self.aicar.rect.y - self.camera.offset_float.y + self.camera.CONST.y)
        self.camera.offset_float.x, self.camera.offset_float.y = int(self.camera.offset_float.x), int(self.camera.offset_float.y)
