from tmo.db import im2k0, im1k4, im2k4, im3k4, im4k4, im5k4, im6k4, tm1k4, tm2k4, pf1k4, pf2k4
import pandas as pd

class Tmo_Alignment:
    
    def __init__(self, curr_im=1):    
        imNk4 = [im2k0, im1k4, im2k4, im3k4, im4k4, im5k4, im6k4, tm1k4, tm2k4]
        #print('im4k4 is not functioning properly and is removed from list')
        #imNk4 = [tm1k4,im1k4, im2k4, im3k4, im5k4]    
        self.imList = imNk4
        self.current_imager = curr_im

    def __repr__(self):
        DF = self._summary()
        return DF.to_string()
        
    def _move_to_target(self, imager_id):
        imager = self.imList[imager_id]
        if ('PPM' in type(imager).__name__) :
            imager.target("YAG1")     
        if ('XPIM' in type(imager).__name__) :
            imager.target("YAG")
        if ('TM' in type(imager).__name__):
            imager.target("Target4")
            
    def _velocity(self, imager, vel=None):
        if type(imager) == int:
            imager = self.imList[imager]
        num_state = imager.target.config.state_count
        for n in range(num_state):
            state = getattr(imager.target.config, ('state0%d' % (n+1)))
            if vel is None:
                print(state.velo.pvname + ': ' + str(state.velo.get()))
            else:
                state.velo.put(vel)
        
    def imager_out(self, imager_id=-1):
        if (imager_id > -1):
            self.current_imager = imager_id
        self.imList[self.current_imager].target("OUT")
        if ('PPM' in type(self.imList[self.current_imager]).__name__) :
            self.imList[self.current_imager].led.set(0)
    
    def next_imager(self):
        self.imager_out(self.current_imager)
        self.current_imager = self.current_imager + 1
        self._move_to_target(self.current_imager)
            
    def prev_imager(self, remove_curr=True):
        if remove_curr:
            self.imager_out(self.current_imager)
        self.current_imager = self.current_imager - 1
        self._move_to_target(self.current_imager)

    def imager_checkout(self, curr_imager):
        self.current_imager = curr_imager
        self._move_to_target(self.current_imager)
        imager = self.imList[self.current_imager]
        if hasattr(imager, 'led'):
            imager.led.set(5.6)
    
    def jump_to_imager(self, imager, remove_curr=True):
        if remove_curr:
            self.imager_out(self.current_imager)
        self.current_imager = imager
        self._move_to_target(self.current_imager)
        
    def _summary(self):
        summary_dict={}
        summary_dict['Imager Name']=[]
        summary_dict['State']=[]
        for ind, im in enumerate(self.imList):
            summary_dict['Imager Name'].append(im.name)
            summary_dict['State'].append(im.target())
        summaryDF = pd.DataFrame(summary_dict)
        return summaryDF
    
    def all_out(self):
        for ind, im in enumerate(self.imList):
            im.target("OUT")
        pf1k4.target("OUT")
        pf2k4.target("OUT")
        
tmo_imagers = Tmo_Alignment()

