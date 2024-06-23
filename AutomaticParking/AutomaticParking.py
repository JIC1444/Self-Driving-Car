import pygame
from Camera import *
from math import *
import cv2

pygame.init()

def rotate(surface: pygame.surface, angle: float, pivot: pygame.math.Vector2, offset: pygame.math.Vector2):
    """
    Args:
        surface: The surface that is to be rotated.
        angle: Rotate by this angle.
        pivot:  The pivot point.
        offset: This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotozoom(surface, -angle, 1)
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.    

class aicar(pygame.sprite.Sprite):
    def __init__(self, image, x, y, angle = 90., acceleration = 0.): #Angle of 90 faces the top of the screen
        super().__init__()
        self.image = image
        self.original_image = image
        self.rect = self.image.get_rect(center=(x, y)) #Change these for the front-wheel steering?
        self.x = x
        self.y = y 
        self.angle = angle
        self.acceleration = acceleration

    def update(self): #Take the coordinates, speed and angle to work out the new coordinates
        self.angle = self.steer()
        self.acceleration = self.accelerate()

        rad_angle = radians(self.angle)
        self.x += self.acceleration * cos(rad_angle) 
        self.y += self.acceleration * sin(rad_angle)

        self.rect.center = (self.x, self.y)
        self.image, self.rect = rotate(self.original_image, self.angle, vec(self.x, self.y), vec(0, 0))


    def steer(self): #Allows the car to steer using the d and a keys
        key = pygame.key.get_pressed()
        if key[pygame.K_d] == True: 
            if self.angle + 1 >= 135: #Won't let the car go past 135˚
                pass
            else:
                self.angle += 1
        elif key[pygame.K_a] == True:
            if self.angle - 1 <= 45: #Won't let the car go past 45˚
                pass
            else:
                self.angle += -1

        """
        elif self.acceleration > 0: #Return steering to 0 slowly if wheels are moving
            if self.angle > 90:
                self.angle -= 0.5
            else:
                self.angle += 0.5
        """
        return self.angle

    def accelerate(self): #Allows the car to stop, go forward and go backward
        key = pygame.key.get_pressed()
        if key[pygame.K_s] == True:
            if self.acceleration + 2 >= 20:
                pass
            else:
                self.acceleration += 2
        elif key[pygame.K_w] == True:
            if self.acceleration -2 <= -20:
                pass
            else:
                self.acceleration += -2
        else:
            self.acceleration = self.acceleration*0.9 #Friction
            
        
        return self.acceleration
    

class ParkCar(aicar): #Using the aicar controls, have the ai input keys to output the car's movement
    def __init__(self, image, x, y, angle=90, acceleration=0):
        super().__init__(image, x, y, angle, acceleration)

        self.tl, self.tr, self.bl, self.br = find_corners()
        self.hazard_n_tl, self.hazard_n_tr, self.hazard_n_bl, self.hazard_n_br = find_corners()
        self.hazard_s_tl, self.hazard_s_tr, self.hazard_s_bl, self.hazard_s_br = find_corners()

    def find_corners(self, surface) -> tuple:
        rect_width, rect_height, rect_center = surface.rect.width, surface.rect.height, surface.get_rect.center()
        tl = (rect_center[0] + rect_width / 2, rect_center[1] + rect_height / 2)
        tr = (rect_center[0] - rect_width / 2, rect_center[1] + rect_height / 2)
        bl = (rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2)
        br = (rect_center[0] + rect_width / 2, rect_center[1] - rect_height / 2)
        return tl, tr, bl, br

    def calc_distance(self, corner_x: float, corner_y: float, hazard_x: float, hazard_y: float):
        dist = (corner_x - hazard_x) / (corner_y - hazard_y)
        return dist

    def nearest_car(): #Find the two nearest cars to the aicar
        cars_n = [] 
        cars_s = [] #List of the detected sprites, from bottom to top of the screen
        sort_distances = []
        for car in cars_n:
            distance = calc_distance(aicar, car)
            sort_distances.append(distance)
        nearest_car_n_dist = min(sort_distances)

        sort_distances = []
        for car in cars_s:
            distance = calc_distance(aicar, car)
            sort_distances.append(distance)
        nearest_car_s_dist = min(sort_distances)

        return nearest_car_n_dist, nearest_car_s_dist
    
    def anti_collision(self): #Might not need this function just yet.
        nearest_n, nearest_s = nearest_car()
        if nearest_n < nearest_s:
            hazard = nearest_n
        else:
            hazard = nearest_s
        
        aicar_tl, aicar_tr, aicar_bl, aicar_br = find_corners(aicar)
        hazard_tl, hazard_tr, hazard_bl, hazard_br = find_corners(hazard)
        
        return self.angle

    def find_bay(): #Locate a space in the cars which is 1.5x larger than the car's height
        nearest_car_n, nearest_car_s = find_corners()
        if nearest_car_n.y - nearest_car_s.y <= aicar_length*1.5: 
            """
            Take the y coordinates of the bottom of the north car and the top of the south car and compare to required length
            """
            bay = __.get_rect()
            return bay

    def approach_bay(self): #Approach the bay at the correct angle next to a parked car, aligning the rears
        while aicar.center.x != nearest_car_n.center.x:
            self.acceleration = 4
            self.angle = 90.
            

        

    def check_distance(): #Checks the distance between the aicar and the cars infront and behind
        pass

    def reverse_into_bay(): #Does the first two reversing moves of a parallel park
        pass

    def check_parking(): #A function which checks the closeness of the left hand tire to the pavement
        pass

    def shimmy(): #Move the car back and forth until check_parking is validated
        pass




SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
BACKGROUND_COLOR = (255, 255, 255)


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Parking Visualization")

FPS = 60
clock = pygame.time.Clock()

#Initializing car image
aicar_img_path = "SELF DRIVING CAR/aiCar.png"
aicar_img = cv2.imread(aicar_img_path)
aicar_img = pygame.surfarray.make_surface(aicar_img)
aicar = aicar(image = aicar_img, x = SCREEN_WIDTH // 2, y =SCREEN_HEIGHT // 2)

camera = Camera(aicar)
follow = Follow(camera, aicar)
camera.set_method(follow)

all_sprites = pygame.sprite.Group(aicar)

#Event handler
run = True
while run:
    dt = clock.tick(60) / 1000 #Time between each loop iteration (may not need this anymore)
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    camera.up_scroll()
    aicar.update() #Update all sprites

    screen.fill((255, 255, 255)) #Clear screen with white
    all_sprites.draw(screen)  # Draw all sprites

    #screen.blit(aicar.image)
    pygame.display.flip() #Update display

pygame.quit()
