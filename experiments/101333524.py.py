from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from pcdsdevices.gauge import GCCPLC
from tmo.tmo_daq_scans import scan
from tmo.db import dream_gsjn
from tmo.db import RE, bp
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.mpl_plotting import LivePlot, LiveTable


class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['gas_jet_grid_scan']

    ghc_dp1 = Cpt(GCCPLC, 'DREAM:DP1:GHC:01', kind='normal')
    ghc_dp2 = Cpt(GCCPLC, 'DREAM:DP2:GHC:01', kind='normal')
    ghc_dp3 = Cpt(GCCPLC, 'DREAM:DP3:GHC:01', kind='normal')
    ghc_mc = Cpt(GCCPLC, 'DREAM:MC:GHC:01', kind='normal')
    ghc_cat1 = Cpt(GCCPLC, 'DREAM:CAT1:GHC:01', kind='normal')
    ghc_cat2 = Cpt(GCCPLC, 'DREAM:CAT2:GHC:01', kind='normal')
    
   
    def gas_jet_grid_scan (self, x_list, z_list):
        
        XL, ZL = scan.build_outer_list(x_list, z_list)
        dets = [self.ghc_dp1, self.ghc_dp2, self.ghc_dp3, self.ghc_cat1, self.ghc_cat2, self.ghc_mc]
        var = dets
        var.extend([dream_gsjn.gas_nozzle_x, dream_gsjn.gas_nozzle_z])
                
#        RE( bp.count(dets), [LivePlot((self.ghc_cat1.name+'_pressure'))] )
        RE(bp.list_scan(dets, dream_gsjn.gas_nozzle_x, XL, dream_gsjn.gas_nozzle_z, ZL), [LivePlot((self.ghc_cat1.name+'_pressure')), LiveTable(var)])


    def __init__(self):
        name = 'user'
        return super().__init__(name=name)
