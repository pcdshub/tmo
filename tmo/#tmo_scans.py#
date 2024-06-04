from ophyd import EpicsSignal, EpicsSignalRO
import logging
import numpy as np
import time
from tmo.bykiks import bykiks

logger = logging.getLogger(__name__)

def move_and_wait(pv, start, stop, steps, wait):
    """
    Parameters:
    ----------
    pv: str
        The pv you want to scan ('TMO:MISC:EVR:01:TRIG3:TDES')

    start: int/float
        Start value for the scan

    stop: int/float
        Stop value for the scan

    steps: int
        Number of steps in the scan

    wait: int
        Number of seconds to wait at each step
    """
    try:
        scan_var = EpicsSignal(pv)
    except Exception as e:
        logger.warning(f'Unable to get pv: {e}')

    steps = np.linspace(start, stop, steps)
    for step in steps:
        logger.info(f'Setting {pv} to {step}')
        scan_var.put(step)
        time.sleep(wait)

    logger.info('Scan Finished')

def las_scan_JPC(start, stop, delta, dwell=30):
    tgt_time = EpicsSignal('LAS:FS14:VIT:FS_TGT_TIME_DIAL')
    delays = np.arange(start, stop, delta) / 1e6
    while True:
        np.random.shuffle(delays)
        for delay in delays:
            logger.info(f'Setting delay {delay}')
            tgt_time.put(delay)
            time.sleep(0.5)
            logger.info(f'Delay set {tgt_time.get()}')
            time.sleep(dwell)

def imprint_scan(x_motor, x_start, x_stop, x_steps, y_motor, y_start, y_stop, y_steps):
	"""Moves motors in a grid and fires one pulse from bykiks at each position,
	y is slow parameter, x is fast.  So it starts at x_start, y_start, then scans
	through x, then makes a step in y and continues.  You must call ACR for SXR burst mode

	Parameters:
	----------
	x_motor: pcdsdevice EpicsMotor like object
		Example is (mr3k4_kbo.x).  The motor you'd like to scan in x

	x_start: float
		Absolute x motor starting position

	x_stop: float
		Absolute x motor ending position

	x_steps: int
		Number of steps for the x scan

	y_motor: pcdsdevice EpicsMotor like object
		Example is (mr3k4_kbo.y).  The motor you'd like to scan in y

	y_start: float
		Absolute y motor starting position

	y_stop: float
		Absolute y motor ending position

	y_steps: int
		Number of steps for the y scan
	"""
	initial_x_pos = x_motor.user_setpoint.get()
	initial_y_pos = y_motor.user_setpoint.get()
	if not bykiks.burst_mode_enable.get():
		logger.warning('Burst Mode not enabled, call ACR.  Exiting')
		#return
	bykiks.shots_to_burst.put(1)
	x_vals = np.linspace(x_start, x_stop, x_steps)
	y_vals = np.linspace(y_start, y_stop, y_steps)
	for y_val in y_vals:
		y_motor.mv(y_val, wait=True)
		for x_val in x_vals:
			x_motor.mv(x_val, wait=True)
			logger.info('Bursting BYKIKS single shot')
			bykiks.request_burst.put(1)
			time.sleep(0.1)  # Consider how to do with callbacks

	logger.info('Restoring initial positions')
	x_motor.mv(initial_x_pos, wait=True)
	y_motor.mv(initial_y_pos, wait=True)
	logger.info('Motors Restored, finished with scan')

