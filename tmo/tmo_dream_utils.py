from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.epics_motor import SmarAct
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.gauge import GCCPLC
from tmo.tmo_daq_scans import _BuildList
from tmo.db import dream_gsjn, tmo_dream
from tmo.db import RE, bp
from bluesky.callbacks import CallbackBase
from bluesky.callbacks import LiveTable, LivePlot
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import os
import datetime
import h5py
import logging


logger = logging.getLogger(__name__)



class SamplePaddle(BaseInterface, Device):
    tab_component_names = True
    x = Cpt(SmarAct, ':m2', doc='dream sample paddle x', kind='normal')
    y = Cpt(SmarAct, ':m1', doc='dream sample paddle y', kind='normal')
    z  = Cpt(SmarAct, ':m4', doc='dream sample paddle z', kind='normal')
    ret  = Cpt(SmarAct, ':m3', doc='dream sample paddle ret', kind='normal')

tmo_dream_sample_paddle = SamplePaddle('TMO:DREAM:MCS2:01', name='tmo_dream_sample_paddle')

class DreamCoil(BaseInterface, Device):
    tab_component_names=True
    tab_whitelist=['on', 'off']
    
    current = Cpt(EpicsSignal, ':GETCUR', write_pv=':SETCUR', kind='normal',
                  doc='coil current')
    voltage = Cpt(EpicsSignal, ':GETVOLTS', write_pv=':SETVOLTS', kind='normal',
                  doc='coil current')
    calc_resistance = Cpt(EpicsSignalRO, ':CALCRES', kind='normal',
                          doc='calculated resistance')
    calc_power = Cpt(EpicsSignalRO, ':CALCPWR', kind='normal',
                          doc='calculated resistance')
    ps_ctrl = Cpt(EpicsSignal, ':GETONOFF', write_pv=':SETONOFF', kind='normal',
                  doc='on/off switch')
    def on(self):
        self.ps_ctrl.put(0)
    
    def off(self):
        self.ps_ctrl.put(1)

tmo_dream_coil = DreamCoil('TMO:DREAM:COIL:PSI:01', name='tmo_dream_coil')
    
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

class DreamGasJet(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['gas_jet_grid_scan','rga_single_mass_scan',
                     'save_data','analyze_data', 'load_data', 'fit_result']

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
        """
        Collect and plot data for a single mass from the RGA.

        Parameters
        ----------
        num_pts : TYPE, optional
            DESCRIPTION. The default is 30.
        mass : TYPE, optional
            DESCRIPTION. The default is 40.

        Returns
        -------
        None.

        """
        cb_list = [LivePlot('tmo_dream_gas_jet_scan_rga_cat_pressure')]
        self.rga_cat.scan_mass = mass
        RE(bp.count([self.rga_cat], num_pts), cb_list)
        
    def gas_jet_grid_scan (self, x_list, z_list, save=None, dwell=1, repeat=1):
        """
        Scan the x and z position of the tmo_dream.gas_nozzle to find the optimum alignment. 

        Parameters
        ----------
        x_list : List of float
            List of x-positions to scan (functions will build a grid).
        z_list : List of float
            List of z-positions to scane (function will build a grid).
        save : bool, optional
            Boolean to indicate to save the data to an h5. The default is True.
        dwell : int, optional
            Number of readings to take at each location. The default is 1.
        repeat : int, optional
            Number of times to repeat the full scan. The default is 1.

        Returns
        -------
        Results of a 2-D gaussian fit are saved to self.fit_result. 

        """
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
        
        x_0 = tmo_dream.gas_nozzle_x.wm()
        z_0 = tmo_dream.gas_nozzle_z.wm()
        settle_x = tmo_dream.gas_nozzle_x.settle_time
        tmo_dream.gas_nozzle_x.settle_time = 2 if settle_x < 2 else settle_x
        settle_z = tmo_dream.gas_nozzle_z.settle_time
        tmo_dream.gas_nozzle_z.settle_time = 2 if settle_z < 2 else settle_z
        
        fig, ax = plt.subplots(2,4) #create figure
        ax = ax.reshape(8) # make axes a single list
        
        cb = [self._cb]
        for i,det in enumerate(dets):
            for key in det.describe().keys():
                cb.append(LivePlot(key, ax=ax[i]))
            
        var = dets.copy()
        var.extend([tmo_dream.gas_nozzle_x.name, tmo_dream.gas_nozzle_z.name])
        cb.append(LiveTable(var))
        
#        RE( bp.count(dets), [LivePlot((self.ghc_cat1.name+'_pressure'))] )
        RE(bp.list_scan(dets, tmo_dream.gas_nozzle_x, XL, tmo_dream.gas_nozzle_z, ZL), cb)
        
        tmo_dream.gas_nozzle_x.mv(x_0)
        tmo_dream.gas_nozzle_z.mv(z_0)
        tmo_dream.gas_nozzle_x.settle_time = settle_x
        tmo_dream.gas_nozzle_z.settle_time = settle_z
        
        d_name = [d.name for d in dets]
        d_name [-1] = 'tmo_dream_gas_jet_scan_rga_cat_pressure'
        self.fit_result = self.analyze_data(d_name, ['tmo_dream_gas_jet_scan_ghc_dp1_pressure','tmo_dream_gas_jet_scan_ghc_dp2_pressure','tmo_dream_gas_jet_scan_rga_cat_pressure'])

    def gas_jet_y_scan (self, y_list, save=None, dwell=1, repeat=1):
        """
        Scan the tmo_dream.gas_nozzle_y over a range a plot the pressure in various chambers

        Parameters
        ----------
        y_list : TYPE
            DESCRIPTION.
        save : TYPE, optional
            DESCRIPTION. The default is None.
        dwell : TYPE, optional
            DESCRIPTION. The default is 1.
        repeat : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        
        self.save_data = save if save is not None else self.save_data
        self._cb.save_data = self.save_data
        tot = len(y_list) * dwell * repeat
        YL = _BuildList(y_list, scan_type='z', nevents_step=1, tot_events=tot, max_events_run=-1, repeat=dwell)

        dets = [self.gcc_nc.pressure,
                self.ghc_dp1.pressure, 
                self.ghc_dp2.pressure, 
                self.ghc_dp3.pressure, 
                self.ghc_cat1.pressure, 
                self.ghc_cat2.pressure,
                self.ghc_mc.pressure,
                self.rga_cat]
        
        y_0 = tmo_dream.gas_nozzle_y.wm()
        settle_y = tmo_dream.gas_nozzle_y.settle_time
        tmo_dream.gas_nozzle_y.settle_time = 1 if settle_y < 1 else settle_y
        
        
        fig, ax = plt.subplots(2,4) #create figure
        ax = ax.reshape(8) # make axes a single list
        
        cb = [self._cb]
        for i,det in enumerate(dets):
            for key in det.describe().keys():
                cb.append(LivePlot(key, 'tmo_dream_gas_nozzle_y_set', ax=ax[i]))
            
        var = dets.copy()
        var.extend([tmo_dream.gas_nozzle_x.name, tmo_dream.gas_nozzle_z.name])
        cb.append(LiveTable(var))
        
        RE(bp.list_scan(dets, tmo_dream.gas_nozzle_y, YL), cb)
        
        tmo_dream.gas_nozzle_y.mv(y_0)
        tmo_dream.gas_nozzle_y.settle_time = settle_y

        
    def analyze_data(self, dets=['tmo_dream_gas_jet_scan_ghc_dp1_pressure','tmo_dream_gas_jet_scan_ghc_dp2_pressure','tmo_dream_gas_jet_scan_rga_cat_pressure'], fits=None):
        """
        Analyze the gas_jet_grid_scan data. 

        Parameters
        ----------
        dets : list of str, optional
            List of detector names to analyze
            Default: ['tmo_dream_gas_jet_scan_ghc_dp1_pressure','tmo_dream_gas_jet_scan_ghc_dp2_pressure','tmo_dream_gas_jet_scan_rga_cat_pressure']

        fits : list of str, optional
            List of detector names to fit with 2-D gaussian. The default is None.
            ['tmo_dream_gas_jet_scan_ghc_dp1_pressure','tmo_dream_gas_jet_scan_ghc_dp2_pressure','tmo_dream_gas_jet_scan_rga_cat_pressure']

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        def twoD_gauss(xy, A, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = xy
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = offset + A * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)) )
            
            return g.ravel()
        
        x = np.array(self._cb.data['tmo_dream_gas_nozzle_x_set'])
        z = np.array(self._cb.data['tmo_dream_gas_nozzle_z_set'])
        
        nc = int(np.ceil(np.sqrt(len(dets))))
        nr = int(np.ceil(len(dets)/nc))
        fig,ax = plt.subplots(nc, nr)
        ax = ax.reshape(nc*nr)
        
        for i,det in enumerate(dets):
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
                    
            ax[i].pcolormesh(np.unique(z), np.unique(x), result, shading='gouraud')
            ax[i].set_title(det)
            ax[i].set_xlabel('gas_nozzle_z')
            ax[i].set_ylabel('gas_nozzle_x')
            
        fig.tight_layout()
        
        #fitting
        if fits is None:
            return None
        else:
            initial_guess = [5E-9, x.mean(), z.mean(), 0.3, 0.3, 0, 1E-9] # for fitting
            opt_dict={}
            for det in fits:
                i = dets.index(det)
                data = np.array(self._cb.data[det])   
                popt, pcov = opt.curve_fit(twoD_gauss, (x,z), data, p0=initial_guess)    
                ax[i].plot(popt[2], popt[1], 'ko')
                opt_dict[det]={}
                for i,parameter in enumerate(['amplitude', 'x0', 'z0', 'sigma_x', 'sigma_z', 'theta', 'offset']):
                    opt_dict[det].update({parameter:popt[i]})
            return opt_dict
            
    def load_data(self, filename):
        file = os.path.join(self._cb._FILEPATH, filename)
        with h5py.open(file) as hf:
            for key in hf:
                self._cb.data[key] = hf[key]
        
        return None
    
    def __init__(self, **kwargs):
        self.save_data = True
        self._cb = JetScanCallback()
        self.fit_result = None
        return super().__init__(**kwargs)

tmo_dream_gas_jet_scan = DreamGasJet(name='tmo_dream_gas_jet_scan')
