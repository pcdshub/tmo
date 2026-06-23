from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.gauge import GCCPLC
from tmo.tmo_daq_scans import _BuildList
from tmo.db import dream_gsjn
from tmo.db import RE, bp
from bluesky.callbacks import CallbackBase
from bluesky.callbacks import LiveTable, LivePlot
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import h5py
import logging
logger = logging.getLogger(__name__)

class JetScanCallback(CallbackBase):
    def __init__(self, filepath = None):
        self.save_data=True
        self._FILEPATH = filepath if filepath is not None else '/cds/home/opr/tmoopr/jet_scan_data/' 
        if not os.path.exists(self._FILEPATH):
            os.makedirs(self._FILEPATH)
            
    def start(self, doc):
        self._start_doc = doc
        self.data = OrderedDict()
        for i, motor in enumerate(doc['motors']):
            self.data[motor+'_set'] = doc['plan_args']['args'][2*i+1]
        if 'filename' in doc:
            self._f_name = doc['filename']
        else:
            self._f_name = 'jet_scan'
        return
    
    def descriptor(self, doc):
        self._describe_doc = doc
        for val in doc['object_keys'].values():
            data_dict_key = val[0]
            self.data[data_dict_key] = []
            self.data[(data_dict_key+' timestamp')]=[]
#        for key in doc['data_keys'].keys():
#            if key == doc['data_keys'][key]['object_name']:
#                self.data[key]=[]
#                self.data[(key+' timestamp')]=[]
        return
    
    def event(self, doc):
        self._event_doc = doc
        for key in doc['data'].keys():
            if key in self.data:
                if type(doc['data'][key]) is list:
                    self.data[key].extend(doc['data'][key])
                else:
                    self.data[key].append(doc['data'][key])
        for key in doc['timestamps'].keys():
            if key in self.data:
                if type(doc['data'][key]) is list:
                    self.data[(key + ' timestamp')].extend(doc['timestamps'][key])
                else:
                    self.data[(key + ' timestamp')].append(doc['timestamps'][key])       
        return    

    def stop(self, doc):
        self._end_doc = doc
        self._f_name += '_%s.h5' % (datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        #restore initial positions
        if self.save_data:
            with h5py.File(self._FILEPATH + self._f_name, 'w') as hf_dat:
                for key in self.data.keys():
                    hf_dat[key] = self.data[key]  
            logger.info('Saved scan data to %s' % (self._FILEPATH + self._f_name))

class TempRGA(BaseInterface, Device):
    "A temporary class for the RGA."
    
    tab_component_names= True
    
    mass = Cpt(EpicsSignal, ':SingleMassScan', kind='normal')
    pressure = Cpt(EpicsSignalRO, ':SingleMassPressure', kind='normal')
    emission_current = Cpt(EpicsSignal, ':GetEmissionCurrent', write_pv=':SetEmissionCurrent')
    multiplier_on = Cpt(EpicsSignal, ':TurnEMOn')
    multiplier_off = Cpt(EpicsSignal, ':TurnEMOff')
    scan_mass = 40
    
    def describe(self):
        return self.pressure.describe()
    
    def stage(self):
        self.emission_current.put(1.0, wait=True)
        self.multiplier_on.put(1, wait=True)
        self.mass.put(self.scan_mass, wait=True)
        self.pressure.read()
        return [self]
        
    def trigger(self):
        self.mass.put(self.scan_mass, wait=True)
        return self.pressure.trigger()
        
    def read(self):
        return self.pressure.read()

    def unstage(self):
#        self.emission_current.put(0, wait=True)
#        self.multiplier_off.put(1, wait=True)
        return [self]

class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['gas_jet_grid_scan','save_data','analyze_data']

    gcc_nc = Cpt(GCCPLC, 'DREAM:NC:GCC:01', kind='normal')
#   gpi_nc = Cpt(, 'DREAM:NC:GPI:01', kind='normal')
    ghc_dp1 = Cpt(GCCPLC, 'DREAM:DP1:GHC:01', kind='normal')
    ghc_dp2 = Cpt(GCCPLC, 'DREAM:DP2:GHC:01', kind='normal')
    ghc_dp3 = Cpt(GCCPLC, 'DREAM:DP3:GHC:01', kind='normal')
    ghc_mc = Cpt(GCCPLC, 'DREAM:MC:GHC:02', kind='normal')
    ghc_cat1 = Cpt(GCCPLC, 'DREAM:CAT1:GHC:01', kind='normal')
    ghc_cat2 = Cpt(GCCPLC, 'DREAM:CAT2:GHC:01', kind='normal')
    rga_cat = Cpt(TempRGA, 'TMO:DREAM:RGA:02', kind='normal')
       
    def rga_single_mass_scan(self, num_pts=30, mass=40):
        cb_list = [LivePlot('user_rga_cat_pressure')]
        self.rga_cat.scan_mass = mass
        RE(bp.count([self.rga_cat], num_pts), cb_list)
        
    def gas_jet_grid_scan (self, x_list, z_list, save=None, dwell=1, repeat=1):
        
        self.save_data = save if save is not None else self.save_data
        self._cb.save_data = self.save_data
        tot = len(x_list) * len(z_list) * dwell * repeat
        XL = _BuildList(x_list, scan_type='z', nevents_step=1, tot_events=tot, max_events_run=-1, repeat=dwell)
        ZL = _BuildList(z_list, scan_type='z', nevents_step=1, tot_events=tot, max_events_run=-1, repeat=len(x_list)*dwell)
#        XL, ZL = scan.build_outer_list(x_list, z_list, scan_type='z' )

        dets = [self.gcc_nc.pressure,
                self.ghc_dp1.pressure, 
                self.ghc_dp2.pressure, 
                self.ghc_dp3.pressure, 
                self.ghc_cat1.pressure, 
                self.ghc_cat2.pressure,
                self.ghc_mc.pressure,
                self.rga_cat]
        
        x_0 = dream_gsjn.gas_nozzle_x.wm()
        z_0 = dream_gsjn.gas_nozzle_z.wm()
              
        cb = [self._cb]
        for det in dets:
            for key in det.describe().keys():
                cb.append(LivePlot(key))
            
        var = dets.copy()
        var.extend([dream_gsjn.gas_nozzle_x.name, dream_gsjn.gas_nozzle_z.name])
        cb.append(LiveTable(var))
        
#        RE( bp.count(dets), [LivePlot((self.ghc_cat1.name+'_pressure'))] )
        RE(bp.list_scan(dets, dream_gsjn.gas_nozzle_x, XL, dream_gsjn.gas_nozzle_z, ZL), cb)
        
        dream_gsjn.gas_nozzle_x.mv(x_0)
        dream_gsjn.gas_nozzle_z.mv(z_0)
        self.analyze_data(self.ghc_dp3.pressure.name)

    def analyze_data(self, det):
        x = np.array(self._cb.data['dream_gsjn_gas_nozzle_x_set'])
        z = np.array(self._cb.data['dream_gsjn_gas_nozzle_z_set'])
        data = np.array(self._cb.data[det])
        result = []
        for x_i in np.unique(x):
            arg_x = np.argwhere(x==x_i)
            ret = data[arg_x]
            z_i = z[arg_x]
            for z_j in np.unique(z):
                arg_z = np.argwhere(z_i==z_j)
                result.append(np.mean(ret[arg_z]))
                
        result = np.reshape(np.array(result), (len(np.unique(x)),len(np.unique(z))) )
        plt.figure()
        plt.pcolormesh(np.unique(x), np.unique(z), result)
        plt.xlabel('gas_nozzle_x')
        plt.ylabel('gas_nozzle_z')

    def __init__(self):
        self.save_data = True
        self._cb = JetScanCallback()
        name = 'user'
        return super().__init__(name=name)
