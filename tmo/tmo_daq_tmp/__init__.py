import logging
from types import SimpleNamespace
import socket
import subprocess 

from .DaqControl import DaqControl
from .BlueskyScan import BlueskyScan
from .TimedRun import TimedRun


logger = logging.getLogger(__name__)


def get_daq(bluesky=False):
    # Construct the args that DaqScan is looking for
    args = SimpleNamespace(
        C='drp-neh-ctl001', # DRP host for control
        v=True,             # Verbosity
        B='DAQ:NEH',        # PV Base
        detname='scan',     # Detector name
        scantype='scan',    # scan type
        g=36,               # Bit mask of readout groups <--???
        c=10,               # Events per step
        p=0,                # Platform
        x=0,                # Master XPM
        t=1000,             # Connection timeout
        )

    #figure out if the DAQ is running (on the local machine)
    thishost = socket.gethostname()
    wd = subprocess.Popen(['wheredaq'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = wd.communicate()
    if str(stdout).find('not running')>=0:
        print('DAQ is currently not running')
        return None
    if str(stdout).find(thishost)<0:
        print('DAQ is not running on this host')
        return None
        
    #figure out of socket is already used.
    #also: 'unbind' socket?

    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    #snelson -- not sure if we need this?
    instr = control.getInstrument()
    if instr is None:
        err = 'Failed to connect to LCLS-II DAQ'
        logger.error(err)
        return

    daqState = control.getState()
    logging.info('initial state: %s' % daqState)
    if daqState == 'error':
        err = 'Daq is in error state'
        logger.error(err)
        return

    if bluesky:
        daq = BlueskyScan(control, daqState=daqState, args=args)

        # Hack over nabs for now to give us the lcls2 daq instead of lcls1
        def _get_daq():
            return daq

        import nabs.preprocessors
        nabs.preprocessors._get_daq = _get_daq

        try:
            from tmo.db import scan_pvs
            #scan_pvs.disable()
        except ImportError:
            pass # Not in full hutch python session

    else:
        daq = TimedRun(control, daqState=daqState, args=args)

    return daq
