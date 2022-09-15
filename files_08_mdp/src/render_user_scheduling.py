import time
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pygame as pg
import pyscreenshot as ImageGrab
import imageio
#from beamforming_calculation import AnalogBeamformer

SLEEP_TIME = 0.3 #time to sleep and allow visualizing the world :)

class Scheduling_RL_render:
    def __init__(self, should_render = False):
        self.should_render = should_render

        # ---- Definindo algumas variáveis que serão usadas na renderização --- 
        
        SLEEP_TIME = 0.5  #time to sleep and allow visualizing the world :)

        self.Rx_position = (0,0)
        self.Rx2_position = (5,5)
        self.scheduled_user = 0
        
        #Fixed objects, which do not move
        self.Tx = [0,0] #adotando o canto inferior esquerdo como referência
        # self.wall1 = [3,4]
        # self.wall2 = [4,4]

          # create discrete colormap
        cmap = colors.ListedColormap(['gray','red', 'green', 'blue'])
        cmap.set_bad(color='w', alpha=0)
        fig = plt.figure()
        self.pg = pg
        self.pg.init()
        self.screen = pg.display.set_mode((600,600))
        clock = pg.time.Clock()
        back = pg.image.load("./src/figs/grid6x6.png")
        self.back = pg.transform.scale(back, (600,600))
        antenna = pg.image.load("./src/figs/antenna.png").convert_alpha()
        self.antenna = pg.transform.scale(antenna, (40,80))
        carro1 = pg.image.load("./src/figs/carro1.png").convert_alpha()
        self.carro1 = pg.transform.scale(carro1, (80,80))
        carro2 = pg.image.load("./src/figs/carro2.png").convert_alpha()
        self.carro2 = pg.transform.scale(carro2, (80,80))

    def print_debug(self):
        print("Renderizando!!")
       
    def set_positions(self, positions, scheduled_user):
        #positions = self.mimo_RL_Environment.get_UE_positions()
        show_debug_info = False
     
        self.Rx_position = positions[0]
        self.Rx2_position = positions[1]
        self.scheduled_user = scheduled_user # 0 ou 1 

        if show_debug_info:
            print("Rx_position", self.Rx_position)
            print("Rx2_position", self.Rx2_position)
            print("scheduled_user", self.scheduled_user)
    
    def render_back(self):
        self.screen.fill([255,255,255])
        self.screen.blit(self.back,(0,0))

    def render_antenna(self):
        self.screen.blit(self.antenna,(self.Tx[0]*100+30,abs(self.Tx[1]-5)*100+16))

    def render_Rx(self):
        self.screen.blit(self.carro1,(self.Rx_position[0]*100 + 10, abs(self.Rx_position[1]-5)*100 + 10))

    def render_Rx2(self):
        self.screen.blit(self.carro2,(self.Rx2_position[0]*100 + 10, abs(self.Rx2_position[1]-5)*100 + 10))

    def render(self):
        time.sleep(SLEEP_TIME)
        #plot beam
        self.render_back()
        self.render_Rx()
        #Rx2 = carro
        self.render_Rx2()
        self.render_antenna()

        self.pg.display.update()

      

  
# if __name__ == '__main__':
#     env = Scheduling_RL_render(should_render=True)
#     env.print_debug()

