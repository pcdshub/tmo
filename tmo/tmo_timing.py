from tmo.lxt2 import lxt, txt, lxt_ttc, get_timing
from tmo.db import im2k4, tm1k4, lm1k4_qadc, li1k4
#from epics import caget, caput, cainfo
#from elog import HutchELog as elog_temp
#elog = elog_temp.from_registry()
#from tmo.db import elog
import matplotlib.pyplot as plt
import numpy as np
#import time
from scipy.signal import find_peaks
#
threshold_sigma = 6;
CFD_walk = 1;
traces_to_average = 100; 

class TmoTiming():
    '''
    '''
    def __init__(self, threshold_sigma=6, CFD_walk=2, diagnostic=False):
        self.sig_avg=None
        self.traces_to_average = 1
        self._ticks_to_ps = 168.2692E-12
        self._threshold = threshold_sigma
        self._CFD_walk = CFD_walk
        self._diag = diagnostic
              
    def _setup_qadc(self, ch):
        '''
        Configure the QADC.
        '''
        lm1k4_qadc.ichan.put(ch%2) #set the channel interleaving mode
        lm1k4_qadc.interleave.put(1) #interleave channels
        lm1k4_qadc.full_en.put(1)  #enable the full output
        lm1k4_qadc.config.put(1) #configure the QADC
        lm1k4_qadc.start.put(1) #Start running the QADC
        
    def _data_cb(self,*args, value, timestamp, **kwargs):
        '''
        Read in the trace, correct the background and average.
        '''        
        trace = np.zeros_like(value)
        self._iter += 1
        for i in range(3):
            trace[i::4] = value[i::4] - value[i::4].mean()
        if self.sig_avg is None:
            self.sig_avg = trace
        else:
            self.sig_avg = (1-(1/self.traces_to_average))*self.sig_avg + (1/self.traces_to_average)*trace

        ts = self._find_edge(-self.sig_avg)
        self._dt.append(ts[0])
        
        #For Plotting        
        if self._diag:
            self._plt_trace.set_ydata(self.sig_avg)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

        return

    def _find_edge(self, trace):
        '''
        Use a CFD to find the edges in the trace
        '''
        tresh = self._threshold * trace.std()
        trace[ np.abs(trace) < tresh ] = 0
        
        ind_peaks, _ = find_peaks(trace, width=4, height=0.001)
        
#        if (np.shape(ind_peaks) != 2):
        if (np.shape(ind_peaks) != 1):
            print('Peak Finding Failed!')
            return 0,0,0
        
        CFD = trace - np.roll(trace, self._CFD_walk)
        x = ind_peaks[0] + np.arange(-2,self._CFD_walk+2)
        p = np.polyfit(x, CFD[x], 3)
        t1 = np.roots(p)[np.isreal(np.roots(p))][-1]
        x = ind_peaks[1] + np.arange(-5,self._CFD_walk)
        p = np.polyfit(x, CFD[x], 3)
        t2 = np.roots(p)[np.isreal(np.roots(p))][-1]
        dt = (t2-t1)*self._ticks_to_ps
        return dt,t1,t2
#        return 0,t1,0
    
    def atm_coarse_timing(self, traces_to_average):
        '''
        Find the Coarse Timing using the diode on TM1K4
        
        Parameters
        ----------
        traces_to_average : int
            Number of traces to average before moving lxt.

        Returns
        -------
        None.

        '''
        self.traces_to_average = traces_to_average
        self._setup_qadc(ch=0)
        
        tm1k4.target('Diode', wait=True)
        
        original_timing = lxt.position
        lxt.mvr(2E-9) # displace the laser 2 ns from the x-rays
        
         
        #Start collecting and averaging data
        if self._diag:
            plt.ion()
            self._fig = plt.figure()
            ax = self._fig.add_subplot(111)
            self._plt_trace, = ax.plot(np.zeros(lm1k4_qadc.length.get()))
            plt.xlim([2720,2740])
            plt.ylim([-0.01,0.001])
  
        
        self._iter = 0
        self._dt = []
    
        uid = lm1k4_qadc.sparsification.out1.subscribe(self._data_cb)
#        uid = lm1k4_qadc.ch2.subscribe(self._data_cb)
        while (self._iter < traces_to_average):
            continue
        lm1k4_qadc.sparsification.out1.unsubscribe(uid)
 #       lm1k4_qadc.ch2.unsubscribe(uid)
                  
        dt_a = np.array(self._dt)
        dt = dt_a[dt_a>0].mean()       

        #move laser
        if dt > 0:
            lxt.mvr(-dt)
            lxt.set_current_position(0)
        else:
            lxt.mv(original_timing)
            
        #print to elog
        get_timing(log=True, msg='Coarse Timing on ATM diode.')

    def ip_coarse_timing(self, traces_to_average):
        
        self.traces_to_average = traces_to_average
        self._setup_qadc(ch=1)
        
        li1k4.mirror.target('Target1', wait=True)
        
        original_timing = lxt.position
        lxt.mvr(2E-9) # displace the laser 2 ns from the x-rays
        
        #Start collecting and averaging data
        if self._diag:
            plt.ion()
            self._fig = plt.figure()
            ax = self._fig.add_subplot(111)
            self._plt_trace, = ax.plot(np.zeros(lm1k4_qadc.length.get()))
            plt.xlim([2720,2740])
            plt.ylim([-0.01,0.001])
  
        self._iter = 0
        self._dt = []
    
        uid = lm1k4_qadc.sparsification.sparse0.subscribe(self._data_cb)
            
tmo_timing=TmoTiming(diagnostic=True)
