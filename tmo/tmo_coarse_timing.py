from tmo.tmo_laser import tmo_laser
#from tmo.lxt2 import txt
from tmo.db import RE, bp, elog
import lmfit
from bluesky.callbacks import LiveTable, LiveFit
from bluesky.callbacks.mpl_plotting import LiveFitPlot, LivePlot
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

  

def qadc_coarse_timing(nevents_step, ch, ip=1, motor=None, scan_range=None, avg_type='MEAN'):
    """
    Functio for coarse timing using the qadc. This function can just plot the measured difference, or it can scan a motor and fit the result with a line.

    Parameters
    ----------
    nevents_step : int
        Number of points to average.
    ch : int
        Channel of qadc to use for averaging (0 or 1).
    ip : int; optional
        Detector to use lm1k4_qadc if ip=1, lm2k4_qadc if ip=2.
    motor : motor device; optional
        Motor to scan (lxt or txt).
    scan_range : list of float; optional
        Range to scan the motor. If no motor is passed, the scan_range should be an integer for the number of points to measure.
    avg_type : str
        Type of average to use for plotting, should be 'MEAN' or 'MEDIAN'

    Returns
    -------
    Fit result : Dictoinary
    
    """
    #Define the function to fit to the data
    def line(x, a, b):
        #return a*x+b
        return -abs(a*x+b)
    #create an lmfit model for the fit with an initial guess
    model = lmfit.Model(line)
    init_guess = {'a':1E9, 'b':0}
    
    #setup detector
    if ip == 1:
        det = tmo_laser.ip1.lm1k4_qadc
    elif ip == 2:
        det = tmo_laser.ip2.lm2k4_qadc 
    else:
        logger.error('IP not recognizied')
        return None
    
    det.avg_config(events=nevents_step, channel=ch)
    
    #define strings for plotting functions
    det_str = ('%s_EDGE_DIFF_%s' % (det.name, avg_type.upper()) )
    det_mean_str = ('%s_EDGE_DIFF_MEAN' % (det.name) )
    det_median_str = ('%s_EDGE_DIFF_MEDIAN' % (det.name) )
    
    if motor is None: #Not a scan
        lt = LiveTable([det_mean_str, det_median_str])
        lp = LivePlot(det_str, marker='o', linestyle='none')
        
        RE( bp.count([det], scan_range), [lt,lp])
        
        return None
    
    else: #is a scan
    
        #which motor
        if 'ip1' in motor.name:
            x_str = motor.name + '_delay'
        elif 'ip2' in motor.name:
            x_str = motor.name + '_delay'
        else: #lxt is passed
            x_str = motor.name + '_setpoint'
    
        lt = LiveTable([x_str, det_mean_str, det_median_str])
        
        #configure a LiveFit object to do the plotting and fitting
        lf = LiveFit(model, det_str, {'x':x_str}, init_guess)
            
        #configure plots for the data
        fig, ax = plt.subplots() #create figure
        lfp = LiveFitPlot(lf, ax=ax, color='r')
        lp = LivePlot(det_str, x_str, ax=ax, marker='o', linestyle='none')
        
        #Use blue sky to do the scan and fit. 
        RE( bp.list_scan([det], motor, scan_range), [lt, lp, lfp])
           
        return lf.result