from tmo.db import RE, bp, elog
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.mpl_plotting import LivePlot
from bluesky.plan_stubs import monitor, unmonitor, open_run, close_run, sleep
from bluesky.preprocessors import monitor_during_wrapper, fly_during_wrapper
from collections import OrderedDict
import datetime
import os
import h5py
import logging
logger = logging.getLogger(__name__)

class SaveCallback(CallbackBase):
    def __init__(self):
        return
               
    def start(self, doc):
        self._start_doc = doc
        self.data = OrderedDict()
        
        self.msg = doc.get('purpose', None)
        self.device = doc['device'] 
        self.filepath = doc.get('filepath', ('/cds/home/opr/tmoopr/%s_data/' % self.device)  )
        self.filename = doc.get('filename', ('%s_%s.h5' % (self.device, datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") ) ) )
          
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)                          
        
        return
    
    def descriptor(self, doc):
        self._describe_doc = doc
        for key in doc['data_keys'].keys():
            self.data[key]=[]
            self.data[(key+' timestamp')]=[]
        return
    
    def event(self, doc):
        self._event_doc = doc
        for key in doc['data'].keys():
            self.data[key].append(doc['data'][key])
            self.data[(key + ' timestamp')].append(doc['timestamps'][key])                           
        return

    def stop(self, doc):
        self._end_doc = doc
        
        if self.msg is None: self.msg=''
        with h5py.File(self.filepath + self.filename, 'w') as hf_dat:
            for key in self.data.keys():
                hf_dat[key] = self.data[key]        
            
        self.msg += '\nSaved data for %s to %s' % (self.device, (self.filepath + self.filename))        
        self.msg += '\nscan exit status: %s' % doc['exit_status']
        logger.info(self.msg) 
        elog.post(msg=self.msg, tags='tmo_record_it')
        return
    
class TmoRecorder():
    
    def __init__(self):
        self._cb = SaveCallback()
        return
            
    def record_data(self, device, num_samples, msg=None, filepath=None, filename=None):
        """
        record the data for a specific device.

        Parameters
        ----------
        device : device
            Device to measure.
        num_samples : int
            Number of samples to record.
        msg : str, optional
            Optional string for elog post.

        """
        det = [device]
        self._md=OrderedDict()
        self._md.update({'device': device.name,
                          'purpose': msg })
        if filename is not None: self._md.update({'filename':filename})
        if filepath is not None: self._md.update({'filepath':filepath})
        
        def monitor_devce(device, duration, md=None):
            yield from open_run(md)  # optional metadata
            yield from monitor(device, name="device_monitor")
            yield from sleep(duration)  # Wait for readings to accumulate.
            yield from unmonitor(device)
            yield from close_run()
        
        RE(bp.count(det,num=num_samples, md=self._md), [self._cb])
#        RE(monitor_device(det,duration=num_samples, md=self._md), [self._cb])
        
    @property 
    def results(self):
        return self._cb.data

tmo_recorder = TmoRecorder()