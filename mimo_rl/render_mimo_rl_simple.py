'''
Shows the grid world using pygame.
Globecom Tutorial - December 7, 2021
Tutorial 29: Machine Learning for MIMO Systems with Large Arrays
Nuria Gonzalez-Prelcic (NCSU),
Aldebaro Klautau (UFPA) and
Robert W. Heath Jr. (NCSU)
'''
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pygame as pg
import pyscreenshot as ImageGrab
import imageio
#from beamforming_calculation import AnalogBeamformer

SLEEP_TIME = 0.3 #time to sleep and allow visualizing the world :)

class Mimo_RL_render:
    def __init__(self, analogBeamformer):
        self.should_save_images_as_gif = False #Not working: enables saving images in the end
        self.analogBeamformer = analogBeamformer
        self.Rx_position = (0,0)
        self.Rx2_position = (5,5)
        self.scheduled_user = 0
        self.beam_index = 0

        #Fixed objects, which do not move
        self.Tx = [1,2]
        self.wall1 = [3,4]
        self.wall2 = [4,4]

        # create discrete colormap
        cmap = colors.ListedColormap(['gray','red', 'green', 'blue'])
        cmap.set_bad(color='w', alpha=0)
        fig = plt.figure()
        self.pg = pg
        self.pg.init()
        self.screen = pg.display.set_mode((600,600))
        clock = pg.time.Clock()
        back = pg.image.load("./figs/grid6x6.png")
        self.back = pg.transform.scale(back, (600,600))
        antenna = pg.image.load("./figs/antenna.png").convert_alpha()
        self.antenna = pg.transform.scale(antenna, (40,80))
        wall_image = pg.image.load("./figs/wall.png").convert_alpha()
        self.wall_image = pg.transform.scale(wall_image, (90,90))
        carro1 = pg.image.load("./figs/carro1.png").convert_alpha()
        self.carro1 = pg.transform.scale(carro1, (80,80))
        carro2 = pg.image.load("./figs/carro2.png").convert_alpha()
        self.carro2 = pg.transform.scale(carro2, (80,80))
        if self.should_save_images_as_gif:
            self.images = []

    def set_positions(self, positions, scheduled_user, beam_index):
        #positions = self.mimo_RL_Environment.get_UE_positions()
        self.Rx_position = positions[0]
        self.Rx2_position = positions[1]
        self.scheduled_user = scheduled_user
        self.beam_index = beam_index

    def plot_beam(self, scheduled_user, beam_index):
        fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
        if(scheduled_user==0):
            colorCar = 'blue'
        else: #scheduled_user==1
            colorCar = 'red'
        angles = self.analogBeamformer.angles_for_plotting
        beam_values = self.analogBeamformer.beams_for_plotting[:,beam_index]
        beam_values = np.abs(beam_values) #take absolute values

        ax2.plot(angles, beam_values, color=colorCar)
        ax2.set_axis_off()
        ax2.grid(False)
        plt.savefig('chosen_beam.png', transparent=True, bbox_inches='tight')

    def render_back(self):
        self.screen.fill([255,255,255])
        self.screen.blit(self.back,(0,0))

    def render_antenna(self):
        self.screen.blit(self.antenna,(self.Tx[0]*100+30,abs(self.Tx[1]-5)*100+16))

    def render_beams(self):
        bestBeam = pg.image.load("chosen_beam.png").convert_alpha()
        bestBeam = pg.transform.scale(bestBeam, (300,300))
        self.screen.blit(bestBeam,(self.Tx[0]*100-100,abs(self.Tx[1]-5)*100-100))

    def render_wall1(self):
        self.screen.blit(self.wall_image,(self.wall1[0]*100 + 5,abs(self.wall1[1]-5)*100 + 9))

    def render_wall2(self):
        self.screen.blit(self.wall_image,(self.wall2[0]*100 + 5,abs(self.wall2[1]-5)*100 + 9))

    def render_Rx(self):
        self.screen.blit(self.carro1,(self.Rx_position[0]*100 + 10, abs(self.Rx_position[1]-5)*100 + 10))

    def render_Rx2(self):
        self.screen.blit(self.carro2,(self.Rx2_position[0]*100 + 10, abs(self.Rx2_position[1]-5)*100 + 10))

    def render(self):
        time.sleep(SLEEP_TIME)

        #plot beam
        self.plot_beam(self.scheduled_user, self.beam_index)                
        self.render_back()
        self.render_Rx()
        self.render_Rx2()
        self.render_antenna()
        self.render_wall1()
        self.render_wall2()
        self.render_beams()
                
        #plt.pause(1)
        if self.should_save_images_as_gif:
            raise NotImplementedError()
            self.images.append(ImageGrab.grab(bbox=(1960, 1030, 2760, 1830)))

        self.pg.display.update()

    def save_images_as_gif(self, file_name, duration=3):
        raise NotImplementedError()
        gif = imageio.mimsave(file_name, self.images, 'GIF', duration=duration)
        #print(gif)
        print('Wrote file', file_name)