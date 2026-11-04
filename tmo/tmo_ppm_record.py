from tmo.db import RE, bp, elog
from tmo.tmo_optics import get_KB1, get_KB2
from tmo.db import im3k4, im4k4, im5k4, im6k4
from ophyd.signal import EpicsSignal
from ophyd.status import DeviceStatus
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.mpl_plotting import LivePlot
from bluesky.preprocessors import monitor_during_wrapper, fly_during_wrapper
from tmo.tmo_beamline_control import gmd, xgmd, tmo_beamline_cntl
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.device import ObjectComponent as OCpt
from collections import OrderedDict
from tmo.tmo_daq_scans import scan
import numpy as np
import datetime
import time
import os
import h5py
import logging
logger = logging.getLogger(__name__)

class PPMCallback(CallbackBase):
    def __init__(self, filepath = None):
        self.save_data=True
        self._FILEPATH = filepath if filepath is not None else '/cds/home/opr/tmoopr/ppm_data/' 
        if not os.path.exists(self._FILEPATH):
            os.makedirs(self._FILEPATH)
        
    def start(self, doc):
        self._start_doc = doc
        self.data = OrderedDict()
        
        self.msg = doc.get('purpose', None)
        self.deviceName = doc['detector'] 
        self.data['background'] = doc.get('background', '')
        self.data['responsivity'] = doc.get('responsivity','')
        self.data['beam rate'] = doc.get('beam rate', '')
        self.data['transmission'] = doc.get('transmission', '')
        self.data['nominal photon energy'] = doc.get('nominal photon energy', '')
        self.data['nominal average power'] = doc.get('nominal average power', '')
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
    
    def event_page(self, doc):
        self._event_page_doc = doc
    
    def stop(self, doc):
        self._end_doc = doc
        self._f_name += '_%s.h5' % (datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        if self.msg is None: self.msg=''
        if self.save_data:
            with h5py.File(self._FILEPATH + self._f_name, 'w') as hf_dat:
                for key in self.data.keys():
                    hf_dat[key] = self.data[key]        
            self.msg += '\nSaved ppm data for %s to %s' % (self.deviceName, (self._FILEPATH + self._f_name))        
            self.msg += '\nscan exit status: %s' % doc['exit_status']
            logger.info(self.msg) 
            self.msg += '\nbeam rate: %s \ntransmission: %s \nnominal photon energy: %s\nnominal avg pwr: %s' % (self.data['beam rate'], self.data['transmission'], self.data['nominal photon energy'], self.data['nominal average power'])
            elog.post(msg=self.msg, tags='ppm_data')
        return

class TmoPpm(BaseInterface, Device):
    im3k4 = OCpt(im3k4.power_meter)
    im4k4 = OCpt(im4k4.power_meter)
    im5k4 = OCpt(im5k4.power_meter)
    im6k4 = OCpt(im6k4.power_meter)
    
    def __init__(self, det_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = getattr(self, det_name)
        self._acquiring=False
        self._paused=False
        self._fly_data={}
        return
    
    def describe(self):
        return self.detector.calibrated_mj.describe()
    
    def stage(self):
        return self.detector.calibrated_mj.stage()

    def trigger(self):
        def mark_done(self, old_value, value, timestamp, **kwargs):
            if type(old_value) == float:
                status.set_finished()
            return
        status = DeviceStatus(self.detector)
        self.detector.calibrated_mj.subscribe(mark_done)
        self.detector.calibrated_mj.trigger()
        return status
    
    def read(self):
        self.detector.calibrated_mj.unsubscribe_all()
        return self.detector.calibrated_mj.read()
    
    def unstage(self):
        return self.detector.calibrated_mj.unstage()
    
    def _data_cb(self, value, timestamp, **kwargs):
        obj = kwargs['obj']
        self.data[obj.name]['value'].extend(value)
        self.data[obj.name]['timestamp'].append(timestamp)
        
    def kickoff(self):
        if self._acquiring:
            raise RuntimeError("Kicking off a second time?!")      
        self._fly_data={}
        self._fly_data[self.detector.calibrated_mj_buffer.name]={'value':[], 'timestamp':[]}
        self.detector.calibrated_mj_buffer.subscribe(self._data_cb)
        self._start_time = time.time()
        self._acquiring = True
        self._paused = False
        
        kickoff_st = DeviceStatus(device=self)
        kickoff_st.set_finished()
        return kickoff_st
    
    def pause(self):
        '''Pause acquisition'''
        logger.debug('called .pause')
        if not self._acquiring:
            return
        self.detector.calibrated_mj_buffer.unsubscribe_all()
        self._paused = True
    
    def resume(self):
        '''Resume acquisition'''
        logger.debug('called .resume')
        if not self._acquiring:
            return
        if not self._paused:
            return
        self.detector.calibrated_mj_buffer.subscribe(self._data_cb)
        self._paused = False
        
    def complete(self):
        if not self._acquiring:
            raise RuntimeError("No collection in progress")
        self.detector.calibrated_mj_buffer.unsubscribe_all()
        self._acquiring = False #not acquiring
        self._paused = False #not paused
        
        completion_status = DeviceStatus(self)
        completion_status.set_finished()
        return completion_status
            
    def describe_collect(self):
        return self.detector.calibrated_mj_buffer.describe()
            
    def collect(self):
        if self._acquiring:
            raise RuntimeError('Acquisition still in progress. Call complete() first.')
        timestamps={}
        value={}
        for attr, data in self._fly_data.items():
            timestamps[attr]=data['timestamp']
            value[attr]=data['value']
        yield dict(time=self._start_time,
            timestamps=timestamps,
            data=value)
    
class PPM_recorder():
    """
    Class for recording data from the power meters. 
    """
    def __init__(self):
        self.save_data = True
        self._cb = PPMCallback()
        self._md=OrderedDict()
        self._kb1_transmission = EpicsSignal("TMO:KBO1:Transmisstion", name='kb1_trans')
        self._kb2_transmission = EpicsSignal("TMO:KBO2:Transmisstion", name='kb2_trans')
            
    def record_data(self, ppm, num_samples, msg=None, save_data=None):
        """
        record the data for a specific ppm.

        Parameters
        ----------
        ppm : Imager
            Imager to with powermeter for measurement.
        num_samples : int
            Number of samples to record. The sampling rate is about 40 Hz.
        save_data : bool, optional
            Flag to save data. The default is to do whatever you did last time.

        Returns
        -------
        None.

        """
        if type(ppm) == str:
            ppm = TmoPpm(ppm, name=ppm)
        else:
            ppm = TmoPpm(ppm.name, name=ppm.name)
        self.save_data = save_data if save_data is not None else self.save_data
        det = [ppm]
        fly_list = [gmd, xgmd, ppm]
        self._md.update({'detector': ppm.detector.name,
                       'background': ppm.detector.background_voltage.get(),
                      'responsivity':ppm.detector.responsivity.get(),
                          'purpose': msg,
                        'beam rate': tmo_beamline_cntl.rate.get(),
                     'transmission': tmo_beamline_cntl.transmission.get(),
            'nominal photon energy': tmo_beamline_cntl.photon_eV.get(),
            'nominal average power': tmo_beamline_cntl.avg_power.get() })
        self._cb.save_data = self.save_data
        
#        RE(monitor_during_wrapper(bp.count(det_count, num=num_samples, md=self._md), det_monitor), [LivePlot(ppm.power_meter.calibrated_mj.name), self._cb])
        RE(fly_during_wrapper(bp.count(det, num=num_samples, md=self._md), fly_list), [LivePlot(ppm.detector.calibrated_mj.name), self._cb])
#        RE(bp.count(det,num=num_samples, md=self._md), [LivePlot(ppm.detector.calibrated_mj.name), self._cb])
      
    def record_with_daq(self, ppm, num_samples):
        scan.untimedRun(record=True)
        self.record_data(ppm, num_samples, save_data=True)
        scan.stopRun()
        return
    
    @property
    def last_average(self):
        avg=None
        for key in self._cb.data.keys():
            if 'timestamp' in key:
                continue
            elif 'buffer' in key:
                continue
            elif 'calibrated_mj' in key:
                avg = np.mean(self._cb.data.get(key))
        return avg
     
    def check_mirror(self,num_samples, kb_pair=1):
        #ppms = {1:[im3k4, im4k4],2:[im5k4,im6k4]}
        ppms = {1:[im3k4, im4k4],2:[im4k4,im6k4]}
        if tmo_beamline_cntl.transmission.get() < 0.98:
            logger.error('Change to full transmission and try again.')
            return
        #check rate
        ori_rate = tmo_beamline_cntl.rate.get()
        #move first ppm
        ppms[kb_pair][0].target('POWERMETER')
        ppms[kb_pair][1].target('POWERMETER', wait=True)
        #check for rate
        new_rate = tmo_beamline_cntl.rate.get()
        logger.info('waiting for rate')
        while new_rate < 0.9*ori_rate:
            new_rate = tmo_beamline_cntl.rate.get()
        time.sleep(10)
        #measure
        logger.info('recording upstream imager data.')
        self.record_data(ppms[kb_pair][0], num_samples, save_data=True)
        I0 = self.last_average
        #remove first ppm and move second ppm
        ppms[kb_pair][0].target('OUT', wait=True)
        #check for rate
        logger.info('waiting for rate')
        new_rate = tmo_beamline_cntl.rate.get()
        while new_rate < 0.9*ori_rate:
            new_rate = tmo_beamline_cntl.rate.get()
        time.sleep(10)
        #measure second ppm
        logger.info('recording downstream imager data.')
        self.record_data(ppms[kb_pair][1], num_samples, save_data=True)
        I1 = self.last_average
        #remove second ppm
        ppms[kb_pair][1].target('OUT')
        #post results.
        trans = I1/I0
        msg = '%s transmission measured for KB%s' %(trans, kb_pair)
        logger.info(msg)
        if kb_pair == 1:
            get_KB1(log=True, msg=msg)
            if self._kb1_transmission.connected:
                self._kb1_transmission.put(trans)
            else:
                logger.error('Transmission not logged, PV disconnected')
        elif kb_pair == 2:
            get_KB2(log=True, msg=msg)
            if self._kb2_transmission.connected:
                self._kb2_transmission.put(trans)
            else:
                logger.error('Transmission not logged, PV disconnected')
        
ppm_recorder = PPM_recorder()    
