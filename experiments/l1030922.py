from tmo.db import lm1k4_qadc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import logging
import sys
import time
logger = logging.getLogger(__name__)
from tmo.tmo_beamline_control import tmo_beamline_cntl



class  QadcMonitor():
    '''
    '''
    def __init__(self, threshold=8):
        self._ticks_to_ps = 168.2692E-12
        self.threshold = threshold
        self._trace = []
              
    def _setup_qadc(self, ch):
        """ Configure the QADC. """
        lm1k4_qadc.ichan.put(ch%2) #set the channel interleaving mode
        lm1k4_qadc.interleave.put(1) #interleave channels
        lm1k4_qadc.full_en.put(1)  #enable the full output
        lm1k4_qadc.config.put(1) #configure the QADC
        lm1k4_qadc.start.put(1) #Start running the QADC
        
    
    def _analyze(self,tr):
        """ Analyze the trace."""
        trace = np.zeros_like(tr)
        for i in range(3):
            trace[i::4] = tr[i::4] - tr[i::4].mean()
        self._trace = trace
        if (trace.max() > (self.threshold * trace.std()) ): 
            cfd = trace - np.roll(trace,1)
            x = np.arange(0,3,1)+np.argmax(cfd)
            y=cfd[x]
            p=np.polyfit(x,y,1)
            peak = np.roots(p)
            return peak[0]
        else:
            return None
           
    def monitor(self):
        '''
        Read in the trace, correct the background and average.
        '''
        def _data_cb(*args, value, timestamp, **kwargs):
            pos = self._analyze(value)
            if pos is None:
                logger.error('There is no signal on the QADC.')
            else:
                self.pos = pos
                if not np.isclose(pos, self.pos0, atol=0.5 ):
                    logger.error( ('Timing Jump: %s ns. Was originally %s, now at %s ' % (((pos-self.pos0), self.pos0, self.pos)*self._ticks_to_ps)) )
            return
        
        self._setup_qadc(0)
        trace = lm1k4_qadc.out0.get()
        pos = self._analyze(trace)
        if pos is None:
            logger.error('There is no signal on the QADC')
        else:
            self.pos0 = pos
            sys.stdout.write("\x1b]2; QADC Monitor \x07")
            lm1k4_qadc.out0.subscribe(_data_cb)
            logger.info('use lm1k4_qadc.out.unsubscribeall() to stop monitoring.')
        
class User():
    def __init__(self):
        self.timing_monitor = QadcMonitor()
        
    def set_transmission(self, transmission=1):
        tmo_beamline_cntl.apply_beam_parameter(transmission=0.5*transmission)
        time.sleep(2)
        tmo_beamline_cntl.apply_beam_parameter(transmission=transmission)
        trans = tmo_beamline_cntl.transmission.get()
        while trans < transmission:
            time.sleep(0.5)
            trans = tmo_beamline_cntl.transmission.get()
        at_set = False
        trans_set = transmission
        while not at_set:
            time.sleep(0.5)
            trans_curr = tmo_beamline_cntl.transmission.get()
            if trans_curr > transmission:
                if trans_curr > 1.18*trans_set:
                    trans_set = 1.02*trans_set
                else:
                    trans_set = 0.98*trans_set
            if np.isclose(trans_set, transmission, atol=0.02):
                trans_set = transmission
                at_set = True
                
            tmo_beamline_cntl.apply_beam_parameter(transmission=trans_set)

            
            
        
        