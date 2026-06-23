from tmo.tmo_laser import tmo_laser
from tmo.lxt2 import txt
from tmo.db import RE, bp, elog
import lmfit
from bluesky.callbacks import LiveFit
from bluesky.callbacks.mpl_plotting import LiveFitPlot, LivePlot
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

def mat_func(scan_range=1000e-12, scan_steps = 20, overlap_window=300e-12, samples=400):
    if txt.wm() > 2e-12:
        logger.info('TXT is not zeroed!') # how to exit gracefully with 
    scan_steps_half = np.round(scan_steps/2)
    scan_axis = np.append(np.arange(-scan_range/2, -overlap_window/2, scan_steps_half), np.arange(overlap_window/2, scan_range/2, scan_steps_half))
    for scan_point in scan_axis:
        txt.mv(scan_point)
    return    

def jpc_func(motor, scan_range, nevents_step, ch, avg_type='MEAN'):
    """
    Here is a function to scan a motor over a range specified by scan_range. The results are plotted and fit with a line. 

    Parameters
    ----------
    motor : motor device
        Motor to scan (lxt or txt).
    scan_range : list of float
        Range to scan txt.
    nevents_step : int
        Number of points to average.
    ch : int
        Channel of qadc to use for averaging (0 or 1).

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
    
    #Figure out which detector to use
    if 'ip1' in motor.name:
        det = tmo_laser.ip1.lm1k4_qadc
        x_str = motor.name + '_delay'
    elif 'ip2' in motor.name:
        det = tmo_laser.ip2.lm2k4_qadc
        x_str = motor.name + '_delay'
    else: #lxt is passed
        det = tmo_laser.ip1.lm1k4_qadc
        x_str = motor.name + '_setpoint'
    
    #define strings for plotting functions
    det_str = ('%s_EDGE_DIFF_%s' % (det.name, avg_type.upper()) )
    
    #configure the qadc for the correct channel and number of points to average
    det.avg_config(events=nevents_step, channel=ch)
    
    #configure a LiveFit object to do the plotting and fitting
    lf = LiveFit(model, det_str, {'x':x_str}, init_guess)
        
    #configure plots for the data
    fig, ax = plt.subplots() #create figure
    lfp = LiveFitPlot(lf, ax=ax, color='r')
    lp = LivePlot(det_str, x_str, ax=ax, marker='o', linestyle='none')
    
    #Use blue sky to do the scan and fit. 
    RE( bp.list_scan([det], motor, scan_range), [lp, lfp])
       
    return lf.result
    
