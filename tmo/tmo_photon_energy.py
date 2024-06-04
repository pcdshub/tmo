from pcdsdevices.beam_stats import BeamEnergyRequest
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory
from pcdsdevices.device import ObjectComponent as OCpt

hf_w = BeamEnergyRequest('TMO',name='hf_w',atol=0,skip_small_moves=False,bunch=1)
hf_2w = BeamEnergyRequest('TMO',name='hf_2w',atol=0,skip_small_moves=False,bunch=2)

hf_w_status = BeamEnergyRequest('TMO',name='hf_w',bunch=1,acr_status_suffix='AO801')
hf_2w_status = BeamEnergyRequest('TMO',name='hf_2w',bunch=2,acr_status_suffix='AO802')

class HF_W2W(SyncAxis):
    hf_w = OCpt(hf_w)
    hf_2w = OCpt(hf_2w)
    tab_component_names = True
    scales = {'hf_w': 0.5}
    warn_deadband = 1E-2
    fix_sync_keep_still = 'hf_2w'
    sync_limits = (350, 2E3)

hf_w2w = HF_W2W('', name='hf_w2w')
