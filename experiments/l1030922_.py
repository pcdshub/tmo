from tmo.db import lm1k4_qadc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import logging
import sys
logger = logging.getLogger(__name__)



class  QadcMonitor():
    '''
    '''
    def __init__(self, threshold=6):
        self._ticks_to_ps = 168.2692E-12
        self._threshold = threshold
              
    def _setup_qadc(self, ch):
        """ Configure the QADC. """
        lm1k4_qadc.ichan.put(ch%2) #set the channel interleaving mode
        lm1k4_qadc.interleave.put(1) #interleave channels
        lm1k4_qadc.full_en.put(1)  #enable the full output
        lm1k4_qadc.config.put(1) #configure the QADC
        lm1k4_qadc.start.put(1) #Start running the QADC
        
    
    def _analyze(self,trace_in):
        """ Analyze the trace."""
        trace = np.zeros_like(trace_in)
        for i in range(3):
            trace[i::4] = trace_in[i::4] - trace_in[i::4].mean()
        tresh = self._threshold * trace.std()
        trace[ np.abs(trace) < tresh ] = 0
        ind_peaks, _ = find_peaks(trace, width=4, height=0.001)
        if len(ind_peaks == 0):
            return None
        else: 
            return ind_peaks[0]
        
    def monitor(self):
        '''
        Read in the trace, correct the background and average.
        '''
        def _data_cb(self,*args, value, timestamp, **kwargs):
            pos = self._analyze(value)
            if pos is None:
                logger.error('There is no signal on the QADC.')
            if not np.close(pos, self.pos, 2 ):
                logger.error ()

            return
        sys.stdout.write("\x1b]2; QADC Monitor \x07")
        self._setup_qadc(0)
        trace = lm1k4_qadc.out0.get()
        pos, _ = find_peaks(trace, width=4, height=0.001)
        if len(pos) == 0:
            logger.error('There is no signal on the QADC')
        else:
            lm1k4_qadc.out0.subscribe(_data_cb)
        
class User():
    def __init__(self):
        self.timing_monitor = QadcMonitor()
        