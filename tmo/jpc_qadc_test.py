from tmo.db import RE, bp, elog
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.mpl_plotting import LivePlot
from bluesky.preprocessors import monitor_during_wrapper, fly_during_wrapper
from collections import OrderedDict
from tmo.tmo_laser import tmo_laser
import datetime
import os
import h5py
import logging
logger = logging.getLogger(__name__)

class EdgeCallback(CallbackBase):
    def __init__(self, filepath = None):
        self.save_data=True
        self._FILEPATH = filepath if filepath is not None else '/cds/home/opr/tmoopr/qadc_test_data/' 
        if not os.path.exists(self._FILEPATH):
            os.makedirs(self._FILEPATH)
        
        
    def start(self, doc):
        self._start_doc = doc
        self.data = OrderedDict()
        
        self.msg = doc.get('purpose', None)
        if 'filename' in doc:
            self._f_name = doc['filename']
        else:
            self._f_name = doc['detector'] 
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
            if type(doc['data'][key]) is list:
                self.data[key].extend(doc['data'][key])
            else:
                self.data[key].append(doc['data'][key])
        for key in doc['timestamps'].keys():
            if type(doc['data'][key]) is list:
                self.data[(key + ' timestamp')].extend(doc['timestamps'][key])
            else:
                self.data[(key + ' timestamp')].append(doc['timestamps'][key])       
        return

    def stop(self, doc):
        self._end_doc = doc
        self._f_name += '_%s.h5' % (datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        if self.msg is None: self.msg=''
        if self.save_data:
            with h5py.File(self._FILEPATH + self._f_name, 'w') as hf_dat:
                for key in self.data.keys():
                    hf_dat[key] = self.data[key]        
            self.msg += '\nSaved qadc data for to %s' % ((self._FILEPATH + self._f_name))        
            self.msg += '\nscan exit status: %s' % doc['exit_status']
            logger.info(self.msg) 
        return


class QadcEdgeRecorder():
    """
    Class for recording data from the power meters. 
    """
    def __init__(self):
        self.save_data = True
        self._cb = EdgeCallback()
        self._md=OrderedDict()
            
    def record_data(self, qadc_string='lm1k4', ch=0, edge=1, nevents_file=1000, num_file=1, msg=None, save_data=None):
        
        self.save_data = save_data if save_data is not None else self.save_data
        if not isinstance(ch, list): ch = [ch]
        if not isinstance(edge, list): edge = [edge]
        
        det=[]
        for c in ch:
            for e in edge:
                if qadc_string == 'lm1k4':
                    det.append(getattr(tmo_laser.ip1.lm1k4_qadc, 'out%d_edge%d'% (c, e) ))
                elif qadc_string == 'lm2k4':
                    det.append(getattr(tmo_laser.ip2.lm2k4_qadc, 'out%d_edge%d'% (c, e) ))
        self._md.update({'detector': qadc_string})
       
        def loop_plan(nevents_file, num_file):
            for irep in range(num_file):
                yield from bp.count(det, num=nevents_file, md=self._md)
        RE(loop_plan(nevents_file, num_file), self._cb )
                
qadc_recorder = QadcEdgeRecorder()  