from tmo.db import elog, camviewer, archive
from collections import OrderedDict
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.pseudopos import SyncAxis
import pandas as pd
from tabulate import tabulate
from bluesky.callbacks import CallbackBase
import time
import os
import copy
import subprocess
import glob
import paramiko
import sys
import getpass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import datetime
import h5py
import tmo.db as db
logger = logging.getLogger(__name__)

def get_camViwer_info(cam):
    
    param_dict=OrderedDict()
    pv=':'.join(cam.image2.array_data.pvname.split(':')[0:-1])
    fname = '/cds/home/opr/tmoopr/.yagviewer/%s' % pv
    if not os.path.exists(fname):
        logger.error('camViewer properties do not exist for %s' % cam.name)
        return None
    
    with open(fname, 'r') as f:
        data = f.readlines()
    
    # put key value pair into dictionary
    for lines in data:
        kv = lines.split()
        param_dict.update({kv[0]:kv[1:]})
    
    #Check if the marker positions are set to global. Global positions are stored in a PV
    if bool(param_dict['globmarks'][0]):
        #overwrite the marker position from the file if the markers are global. 
        for i in range(4):
            cord = []
            for d in ['X','Y']:
                m = EpicsSignal( (cam.prefix + ('Cross%s%s_RBV') % (i+1,d) ) )
                cord.append( str(m.get()) )
            param_dict[('m%s'%(i+1))] = cord
    
    return param_dict

def attach_cam(cam_list):
    attachment = []
    for i,cam_name in enumerate(cam_list):
        if hasattr(camviewer, cam_name):  
            cam = getattr(camviewer, cam_name)
            if not cam.connected:
                logger.error('%s not connected' % cam_name)
                continue
            fig=plt.figure(501+i, figsize=(10,10))
            a_size = cam.image2.array_size.get()
            data = cam.image2.array_data.get()
            data = np.resize(data, a_size[1:])
            viewer_data = get_camViwer_info(cam)
            ax = plt.imshow(data, cmap='hot')
            ax.axes.set_title(cam.name)
            
            # if cam_name.lower().startswith('im') and cam_name.lower().endswith('k4'):
            #     Xc = EpicsSignal(cam_name.upper()+':PPM:CAM:X_RTCL_CTR')
            #     Yc = EpicsSignal(cam_name.upper()+':PPM:CAM:Y_RTCL_CTR')
            #     X = EpicsSignal(cam_name.upper()+':PPM:CAM:X_BM_CTR')
            #     Y = EpicsSignal(cam_name.upper()+':PPM:CAM:Y_BM_CTR')
                
            #     plt.plot(Xc.get(), Yc.get(), '+:w', label='RTCL_CTR')
            #     plt.plot(X.get(), Y.get(), '+:g', label='BM_CTR')
            
            if not (viewer_data is None):
                xl=ax.axes.get_xlim()
                yl=ax.axes.get_ylim()
                ax.set_clim((int(viewer_data['colormin'][0]),int(viewer_data['colormax'][0])))
                #add cursors
                plt.plot(int(viewer_data['m1'][0]), int(viewer_data['m1'][1]), '+:b', label=','.join(viewer_data['m1']))
                plt.plot(int(viewer_data['m2'][0]), int(viewer_data['m2'][1]), '+:r', label=','.join(viewer_data['m2']))
                plt.plot(int(viewer_data['m3'][0]), int(viewer_data['m3'][1]), '+:c', label=','.join(viewer_data['m3']))
                plt.plot(int(viewer_data['m4'][0]), int(viewer_data['m4'][1]), '+:m', label=','.join(viewer_data['m4']))
                #add ROI
                roi_rect = plt.Rectangle((int(viewer_data['ROI'][0]), int(viewer_data['ROI'][1])), int(viewer_data['ROI'][2]), int(viewer_data['ROI'][3]), linewidth=1, edgecolor='green', facecolor='none', label=','.join(viewer_data['ROI']))
                plt.gca().add_patch(roi_rect)
                #ax.axes.get_xaxis().set_visible(False)
                #ax.axes.get_yaxis().set_visible(False)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
                #reset axes
                ax.axes.set_xlim(xl)
                ax.axes.set_ylim(yl)
            fig.tight_layout()
            fig.savefig('/cds/home/opr/tmoopr/tmoopr_elog_plots/cam%s.png' % i)
            plt.close(fig)
            attachment.append('/cds/home/opr/tmoopr/tmoopr_elog_plots/cam%s.png' % i)
        else:
            logger.error('%s is not a valid camera name' % cam_name)
    return attachment

def add_tags (tags, tag_to_add):
    """
    Add tags to either list of tags or string of tags

    Parameters
    ----------
    tags : str or list of str
        set of tags to return.
        
    tag_to_add: str
        tag to append to list

    Returns
    -------
    tags : str or list of str
        Concatenated list of tags.

    """
    if isinstance(tags, list):
        tags.append(tag_to_add)
    elif isinstance(tags, str):
        if tags == '':
            tags = tag_to_add
        else:
            tags+=(' ' + tag_to_add) 
    return tags
    
def log_positions(curr_positions, log=False, msg=None, include_cam=None, set_points=None, **kwargs):
    """
    Parameters
    ----------
    log : bool, optional
        Flag to post the status to elog. The default is False.

    msg : str, optional
        Message to include in elog post. The default is None.

    include_cam : list of str, optional
        List of camera names to append to elog post.
        
    run : int, optional
        Associate the post with a specific run

    tags : list, optional
        List of tags to add to the post

    attachments : list, optional
        These can either be entered as the path to each attachment or a
        tuple of a path and description

    facility : bool, optional
        Post to the facility logbook

    experiment: bool, optional
        Post to the experimental logbook

    title : str, optional
        Interprets the message as a HTML message with this as the title

        
    """

    if set_points is None:
        positionDF = pd.DataFrame({"Device:": curr_positions.keys(), "Position:": curr_positions.values()})
    else:
        positionDF = pd.DataFrame({"Device:": curr_positions.keys(), 
                                   "Position:": curr_positions.values(),
                                   "Set Point:": set_points.values()})
#    positionDF_str = positionDF.to_string()
#    positionDF_str = positionDF_str.replace(" ","_")
    positionDF_str = tabulate(positionDF, headers="keys", tablefmt="pretty")
    
    print (positionDF_str)
 
    if (log) or (msg is not None):
        if include_cam is not None:
            if not (type(include_cam) is list):
                include_cam = [include_cam]
            kwargs['attachments'] = kwargs.get('attachments', [])
            attachements = attach_cam(include_cam)
            kwargs['attachments'].extend(attachements)
            kwargs['tags']=add_tags(kwargs.get('tags', ' '), " ".join(include_cam))         
        
        if (msg is not None):
            msg = msg + "\n" + positionDF_str
        else:
            msg = positionDF_str
            kwargs['tags']=add_tags(kwargs.get('tags', ' '), 'script-generated')
#        kwargs['tags']=add_tags(kwargs.get('tags', ' '), 'script-generated')
        elog.post(msg, **kwargs)


def get_cam(cam_name, msg=None, **kwargs):
    """
    Save a camera image to the elog with the properties of the camera as the message. 

    Parameters
    ----------
    cam_name : str
        String for the camera name.

    msg : TYPE, optional
        DESCRIPTION. The default is None.

    """
    
    if hasattr(camviewer, cam_name):
        cam = getattr(camviewer, cam_name)
    else:
        logger.error('cam_name not recognizied.')
        return
    
    params = ['gain', 'acquire_period', 'acquire_time', 'trigger_mode']
    settings=OrderedDict()
    for param in params:
        settings[param] = getattr(cam.cam, param).get()
    
    kwargs['tags']=add_tags(kwargs.get('tags', ' '), 'cam_settings')
    
    log_positions(settings, log=True, msg=msg, include_cam=[cam_name], **kwargs)
        
    
def get_motion(devices, log, msg, include_cam, **kwargs):
    curr_positions = OrderedDict()
    setpoints = OrderedDict()    
    for device in devices:
        for key in dir(device):
            if key == 'position': #a single motor was passed, not a device
                motor = device
            else:
                motor = getattr(device, key)
            if hasattr(motor, 'connected'):
                if motor.connected:
                    if hasattr(motor, 'position'):
                        curr_positions[motor.name] = getattr(motor, 'position')
                        if hasattr(motor, 'setpoint'):
                            setpoints[motor.name] = getattr(motor, 'setpoint').get()
                        elif hasattr(motor, 'user_setpoint'):
                            setpoints[motor.name] = getattr(motor, 'user_setpoint').get()
                        else:
                            setpoints[motor.name] = None 
    log_positions(curr_positions, set_points=setpoints, log=log, msg=msg,  include_cam=include_cam, **kwargs)

def get_device(devices, log=False, msg=None, include_cam=None, **kwargs):
    """
    Log the position for a device or a list of devices. This should work for any device with subdevices that have a "position," "setpoint," or "user_setpoint" method.
     
    """
    if not isinstance(devices, list):
        devices = [devices]
#    kwargs['tags'] = kwargs.get('tags', ' ') + ' lamp_positions'        
    get_motion(devices, log, msg, include_cam, **kwargs) 
get_device.__doc__ += log_positions.__doc__       

def toggle_arb(device, latch=False, log=True):
    """
    Toggle the arbitration state from True to False and back to True for the device.

    Parameters
    ----------
    device : device with atribute target
        Device to have arbitration toggled.
    
    log : bool, optional
        Write to elog that arbitration has been toggled. The default is True.

    latch: bool, optional
        Latch arb_enable to False
    """
    if hasattr(device, 'target'):
        target = getattr(device, 'target')
    elif hasattr(device, 'mirror'):
        target = getattr(device, 'mirror')
    elif hasattr(device, 'coating'):
        target = getattr(device, 'coating')
    elif hasattr(device, 'laser_incoupling'):
        target = getattr(device, 'laser_incoupling')
    elif hasattr(device, 'arb_enable'):
        target = device
    else:
        logger.error(device.name + ' does not have attribute "target" or "arb_enable".')
        return      

    if not hasattr(target, 'arb_enable'):
        logger.error(target.name + ' does not have attribute "arb_enable".')
    else:
        target.arb_enable.put(0)
        if latch:
            msg = input('Reason for by-passing PMPS arbitration: ')
            msg = 'Disable arbitration for '+target.name + '\n' + msg
            elog.post(msg, tags=(device.name+' PMPS override'))
            return
        else:
            time.sleep(2)
            target.arb_enable.put(1)
            if log:
                elog.post('toggled arbitration status of '+target.name, tags=(device.name+' PMPS_toggle'))
            return

def motor_arch_compare(devices, tstamp, plot=False):
    """
    Compare the current position of a device or list of devices to a previous condition from archived data.

    Parameters
    ----------
    devices : device with user_setpoint attribure
        Device or list of devices to compare to archive.

    tstamp : datetime object or list
        date/time for data to retrieve from archiver data.
        
    plot : boolean, optional
        Flag to plot the time-series

    Returns
    -------
    None.

    """
    
    if type(tstamp) is list:
        tstamp = datetime.datetime(*tstamp)
    elif not (type(tstamp) is datetime.datetime):
        logger.error('Arguement is not a valid time stammp')
        return
    
    curr_pos=OrderedDict()
    prev_pos=OrderedDict()
            
    if not isinstance(devices, list):
        devices = [devices]
    
    if plot:
        if len(devices) == 1:
            fig,ax = plt.subplots(1,1)
            ax = [ax]
        else:
            nc = int(np.ceil(np.sqrt(len(devices))))
            nr = int(np.ceil(len(devices)/nc))
            fig,ax = plt.subplots(nc, nr)
            ax = ax.reshape(nc*nr)
        ax_ind=0
        
    for device in devices:
        if hasattr(device, 'user_readback'):
            curr_pos[device.name]=device.position
            pv=device.user_readback.pvname
            if plot: 
                #get lots of data
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=datetime.datetime.now()) 
                arch_data.to_dataframe()[pv].plot(ax=ax[ax_ind])
                ax[ax_ind].set_xlabel('timestamp')
                ax[ax_ind].set_ylabel(pv)
                ax_ind+=1
            else: #get only a little data
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1)) 
            
            prev_pos[device.name]=arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
           
        elif isinstance(device, str):
            dev_sig = EpicsSignal(device, name=device)
            pv=dev_sig.pvname
            curr_pos[dev_sig.name]=dev_sig.get()
            if plot: #get lots of data
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=datetime.datetime.now())
                arch_data.to_dataframe()[pv].plot(ax=ax[ax_ind])
                ax[ax_ind].set_xlabel('timestamp')
                ax[ax_ind].set_ylabel(pv)
                ax_ind+=1
                                    
            else: #get only a little data
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1))
            
            prev_pos[dev_sig.name]=arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
            
            
        else: #device might have sub-devices with user_readback.
            logger.debug('%s does not have "user_readback", checking sub-devices.')
            sub_dev = False
            for key in dir(device):
                mtr = getattr(device, key)
                if hasattr(mtr, 'user_readback'):
                    if mtr.connected:
                        sub_dev=True
                        curr_pos[mtr.name]=mtr.position
                        pv=mtr.user_readback.pvname
                        if plot:
                            if ax_ind>0:
                                nc = int(np.ceil(np.sqrt((ax_ind+1))))
                                nr = int(np.ceil((ax_ind+1)/nc))
                                gs = gridspec.GridSpec(nc, nr)
                                for i in range(ax_ind):
                                    ax[i].set_subplotspec(gs[i])
                                ax.append(fig.add_subplot(nc, nr, ax_ind+1))
                            arch_data = archive.get(pv, xarray=True, start=tstamp, end=datetime.datetime.now())
                            arch_data.to_dataframe()[pv].plot(ax=ax[ax_ind])
                            ax[ax_ind].set_xlabel('timestamp')
                            ax[ax_ind].set_ylabel(pv)
                            ax_ind+=1
                            
                        else:
                            arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1))
                        
                        prev_pos[mtr.name]=arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
            if not sub_dev:
                logger.error('%s does not have a user_readback or sub-devices with user_readback' % device.name)
                return
        
    if plot:
        fig.autofmt_xdate()
        fig.tight_layout()
    settings = {    'Device':curr_pos.keys(),
                   'Current':curr_pos.values(), 
                   'Archive':prev_pos.values(), 
                'Difference':np.array(list(curr_pos.values()))-np.array(list(prev_pos.values()))}            

    settingsDF_str = tabulate(pd.DataFrame(settings), headers="keys", tablefmt="pretty")
    print(settingsDF_str)    
    return

def update_target(device, state_name=None, set_pos=None):
    """"
    Updates the target position of the device. After updating a message is posted to the instrument elog. 
    
    Parameters
    ----------
    device : device with attribute target
        Device with target states to update. The device should have either a .target method or a .config method. 
    
    state_name : str
        Name of the state to update. If no state name is given, a list of current states is returned. 
        
    set_pos : float, optional
        New position for the "state_name" state. If no position is given the current dial_position is used.

    """
    if hasattr(device, 'target'):
        config = device.target.config
    elif hasattr(device, 'config'):
        config = device.config
    else:
        logger.error(device.name + ' does not have a "target" attribute or "config" attribute.')
        return
        
    if state_name is None:
        state_dict = {}
        for count in range(config.state_count):
            state_config = getattr(config, ('state%02g' % (count+1) ) )
            state_dict.update({state_config.state_name.get(): state_config.setpoint.get()})
                    
        log_positions(state_dict, log=False)
        return
        
    else:
        state_num = None
        for count in range(config.state_count):
            state_config = getattr(config, ('state%02g' % (count+1) ) )
            _name = state_config.state_name.get() 
        
            if _name == state_name:
                state_num = count+1
                break

        if state_num is None:
            logger.error((state_name + ' not found in %s states.' % device.name))
            return
     
        state_update={'state_name': state_name, 'state_num':state_num}
        state_update['previous position'] = state_config.setpoint.get()
        
        log = True
        
        set_pos = device.y_motor.dial_position.get() if set_pos is None else set_pos
        state_update['new position'] = set_pos
        state_config.setpoint.put(set_pos)
    
        msg = 'Update target positions for %s %s' % (device.name, state_name)
        log_positions(state_update, log=log, msg=msg, tags=[device.name, 'states_update'], facility=True)
        return        


def update_target_multi(device, state_name=None):
    """"
    Updates the target position of a device with multi-axis states. The state specified in state_num is updated to the current position. 
    After updating a message is posted to the instrument elog. 
    
    Parameters
    ----------
    device : device with target config, e.g. sp1k4.zone_plate
        Device with target states to update.
    
    state_name : str
        Name of the state to update. If no state name is given, a list of current states is returned. 
        
    """

    axes = ['x','y','z']
    
    if not hasattr(device, 'config'):
        logger.error((device.name + ' does not have a config attribute.'))
        return
    
    motor_count = device.config.motor_count
    if not (motor_count>1):
        logger.error((device.name + ' does not have multiple axes.'))

    state_count = device.config.state_count
    
    if state_name is None:
        state_dict = {}
        for count in range(state_count):
            for m in range(motor_count):
                state_config = getattr(device.config, ('m%s_state%02g' % (m+1, count+1) ) )
                state_dict.update({(state_config.state_name.get() + ('_m%s' % (m+1))): state_config.setpoint.get()})
        log_positions(state_dict, log=False)
        return
    
    else: #state name is provided
        state_num = None
        for count in range(state_count):
            state_config = getattr(device.config, ('m1_state%02g' % (count+1) ) )
            _name = state_config.state_name.get() 
        
            if _name == state_name:
                state_num = count+1
                break
    
        if state_num is None:
            logger.error((state_name + ' not found in %s states.' % device.name))
            return        

        names = device.name.split("_")
        parent_dev = getattr(db, names[0])
        motor_prefix = "_".join(names[1:])
        
        state_update={'state_name': state_name, 'state_num':state_num}
        
        for m in range(motor_count):
            state_config = getattr(device.config, ('m%s_state%02g' % (m+1, state_num) ) )
            state_update[('previous m%s'%(m+1))] = state_config.setpoint.get()
            
            motor = getattr(parent_dev, (motor_prefix + "_" + axes[m]))
            set_point = motor.dial_position.get()
            state_update[('new m%s'%(m+1))] = set_point
            state_config.setpoint.put(set_point)
        
        msg = 'Update target positions for %s %s' % (device.name, state_name)
        log_positions(state_update, log=True, msg=msg, tags=[device.name, 'states', 'update'], facility=True)

def add_device_to_arch(device, arch_file = '/cds/home/opr/tmoopr/misc/epicsArch_tmo_exp_specific.txt'):
    """
    Add the device to the DAQ epics Archiver and/or elog post. 

    Parameters
    ----------
    device : Device with pvname or user_readback method
        Device to be added to the archiver. The name for the variable will be taken from the device.name method.

    arch_file : str, optional
        epicsArch file where the PV should be added. This is an optional arguement.  
        The default is '/cds/home/opr/tmoopr/misc/epicsArch_tmo_exp_specific.txt'.

    Returns
    -------
    None.

    """
    
    key = device.name
    if hasattr(device, 'pvname'):
        pv = device.pvname
    elif hasattr(device, 'user_readback'):
        pv = device.user_readback.pvname
    else:
        logger.error('Cannot find PV associated with %s' % key)
        return
    new_lines = '\n#Added with add_device_to_arch \n*%s \n%s ca \n' % (key, pv)
    with open(arch_file, 'a') as file:
        file.write(new_lines)    
    logger.info('added %s to %s as %s' % (pv, arch_file, key))
    return    

class EpicsSignalAlarm(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['get', 'put']
    hihi = Cpt(EpicsSignal, '.HIHI', kind='normal')
    lolo = Cpt(EpicsSignal, '.LOLO', kind='normal')
    high = Cpt(EpicsSignal, '.HIGH', kind='normal')
    low = Cpt(EpicsSignal, '.LOW', kind='normal')
    
    def get(self):
        alarm_dict = {
            'HIHI': self.hihi.get(),
            'HIGH': self.high.get(),
             'LOW': self.low.get(),
            'LOLO': self.lolo.get()            
            }
        log_positions(alarm_dict, log=False)
        
    def put(self, set_points):
        self.hihi.put(set_points[0])
        self.high.put(set_points[1])
        self.low.put(set_points[2])
        self.lolo.put(set_points[3])

def get_ipmi_creds(server):
    path = '/cds/group/pcds/admin/ipmicreds/'
    file = path + server + '-ipmi.ini'
    with open(file) as f:
        f_contents = f.read()
        print(f_contents)
    return        

class Shmem():
    
    def __init__(self, username, version='mbes'):
        pwd = getpass.getpass()
        self.user = username
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect('drp-srcf-mon003', username=username, password=pwd)
        self.version = version
        self._clear_pv = EpicsSignal('TMO:VARS:TTZF_sPost2', name='shmem_clear_pv')
        
        sys.stdout.write("\x1b]2;shmem\x07")
        
        if not ('bash' in self._check_shell()):
            logger.warning('remote shell is not bash')
        
    def _check_shell(self):
        """ Check the shell of s3df"""
        self._stdin_start, self._stdout_start, self._stderr_start = self.ssh.exec_command('echo $SHELL')
        out = self._stdout_start.read().decode('utf-8')
        
        logger.debug(out)
        parts = out.split('/')
        return parts[-1]
        
    def start(self, version=None):
        """
        Starts the shared memory on a single core. To view any output or error use Shmem.output()
        
        Parameters
        ----------
        version : str, optional
            Which version of shmem to run: 'mbes', 'mrco', 'timing'. The default is saved in Shmem.version.
        
        Returns
        -------
        None.

        """
        #update version
        self.version = version if version is not None else self.version
        
        #check shell
        shell = self._check_shell()
        if not ('bash' in shell):
            logger.info('shell is not bash. I will try, but no promises.')
        
        if 'dream' in self.version:
            if 'bash' in shell:
                cmnd = 'cd /cds/home/opr/tmoopr/tmo_ana/config'
                cmnd += '\n source setup.sh'
                cmnd += '\n ana'
            else:
                cmnd = 'cd /cds/home/opr/tmoopr/tmo_ana/config'
                cmnd += '\n bash -c "source setup.sh"'
                cmnd += '\n bash -c "ana &"'
        
        else:
            if 'bash' in shell:
                cmnd = 'source /cds/group/pcds/dist/pds/tmo/scripts/setup_env.sh'
                cmnd += '\n cd /cds/home/opr/tmoopr/tmo_ana/tmo-shmem/%s' % self.version
                cmnd += '\n python online_analysis.py &'
            else:
                cmnd = 'bash -c "source /cds/group/pcds/dist/pds/tmo/scripts/setup_env.sh"'
                cmnd += '\n bash -c "cd /cds/home/opr/tmoopr/tmo_ana/tmo-shmem/%s"' % self.version
                cmnd += '\n bash -c "python online_analysis.py &"'
            
        logger.debug(cmnd)
        self._stdin_start, self._stdout_start, self._stderr_start = self.ssh.exec_command(cmnd)
                
        return 
        
    def start_MPI(self,n_cores=2):
        """
        Starts the shared memory on multiple cores. To view any output or error use Shmem.output()

        Parameters
        ----------
        n_cores : int, optional
            Number of cores to use for shmem. The default is 2.

        Returns
        -------
        None.

        """
        cmnd = 'source /cds/group/pcds/dist/pds/tmo/scripts/setup_env.sh'
        cmnd += '\n cd /cds/home/opr/tmoopr/tmo_ana/tmo-shmem/%s' % self.version
        cmnd += f'\n mpirun -n {n_cores} --mca btl vader,self python online_analysis.py &'
        logger.debug(cmnd)
        self._stdin_start, self._stdout_start, self._stderr_start = self.ssh.exec_command(cmnd)
               
        return
        
    def stop(self, jobid=None):
        """
        Stop the shared memory. This will try to stop the most recent process you launched. 
        If this does not work, try using Shmem.find_job to locate the job_id and try using that as the argument.   

        Parameters
        ----------
        jobid : int, optional
            job_id to kill

        Returns
        -------
        None.

        """
        if jobid is None:
            jobid = self.find_job(display=False)
            if jobid is None:
                logger.error('could not find job easily, try using Shmem.find_job()')
                cmnd = 'kill -9 %1'
            else:
                cmnd = 'kill -9 %s' % jobid
        else:
            cmnd = 'kill -9 %s' % jobid
            
        logger.debug(cmnd)
        self._stdin_stop, self._stdout_stop, self._stderr_stop = self.ssh.exec_command(cmnd)
                
        return
    	           
    def restart(self, version='mbes'):
        """
        Toggle the shared memory to stop and then start again. 
        The method will error if it cannot stop the shmem.

        Parameters
        ----------
        version : str, optional
            Which version of shmem to run: 'mbes', 'mrco', 'timing'. The default is 'mbes'.

        Returns
        -------
        None.

        """
        jobid = self.find_job(display=False)
        if jobid is None:
            logger.error('could not find shmem running')
            return
        
        self._stdin_stop, self._stdout_stop, self._stderr_stop = self.ssh.exec_command(('kill -9 %s' % jobid))
        error_msg = self._stderr_stop.read().decode('utf-8')
        if error_msg != '':
            error_msg+= '\n shmem not restarted'
            logger.error(error_msg)
            return
        
        self.start(version)
        
        
        
    def output(self, phase='start'):
        """
        Display the output from the ssh commands.

        Parameters
        ----------
        phase : str, optional
            Which phase ('start' or 'stop') to view. The default is 'start'.

        Returns
        -------
        None.

        """
        if phase == 'start':
            print('Start output: \n')
            #print(self._stdout_start.read().decode('utf-8'))
            for line in iter(self._stdout_start.readline, ""):
                print(line, end="")
            print('\nErrors: \n')
            print(self._stderr_start.read().decode('utf-8'))
        
        elif phase == 'stop':
            print('Start output: \n')
            print(self._stdout_stop.read().decode('utf-8'))
            print('\nErrors: \n')
            print(self._stderr_stop.read().decode('utf-8'))
            
        else:
            print('phase not recognizied')
        
    def find_job(self, display=True):
        """
        Print a list of jobs running on the server which have the word python in them. 
        It will try to find the right job in the list and suggest an id. 

        Returns
        -------
        jobid : int
            Best guess for the job that you want to kill using Shmem.stop(job_id).

        """
        cmnd = 'ps -ef | grep python'
        logger.debug(cmnd)
        stdin, stdout, stderr = self.ssh.exec_command(cmnd)
        
        out = stdout.read().decode('utf-8')
        if display: print(out)
        
        jobid = None
        for line in out.splitlines():
            ind = line.find('?')
            if 'online_analysis' in line[ind:] :
                parts = line.split()
                jobid = int(parts[1])
            elif 'dream' in line:
                parts = line.split()
                jobid = int(parts[2])
        return jobid
    
    @property    
    def running(self):
        return not(self.find_job(display=False) is None)
    
    def clear_plots(self):
        """Clear the shared memory plots by writing to the clear PV."""
        self._clear_pv.put(1)
        
def daq_crash(msg=None, restart=True, log_error=True, **kwargs):
    
    kwargs['tags']=add_tags(kwargs.get('tags', ' '), 'daq_crash')
    msg = msg if msg is not None else ''
    msg = 'DAQ crashed.\n' + msg
    
    #need to add date functionality...
    if log_error: #load log file
        current_time = datetime.datetime.now()    
        f_path = '/cds/home/opr/tmoopr/%02d/%02d/' % (current_time.year, current_time.month)
        log_files = glob.glob(os.path.join(f_path, "*:control_gui.log"))
        new_file = ''
        ts = 0
        for file in log_files:
            if os.path.getmtime(file) > ts: 
                ts = os.path.getmtime(file)
                new_file = file
        f=open(new_file)
        msg+= '\n'
        for line in iter(f.readline, ""):
            if '<E>' in line:
                ind = line.find('<E>')
                parts = line.split()
                msg+= '%s %s %s' % (parts[0], parts[1], line[ind:])
        f.close()
        
    elog.post(msg, **kwargs)
    
    if restart: subprocess.run(['restartdaq'])
        

class SyncAxisTmo(SyncAxis):
    tab_whitelist = ['set_offset_to_current_position']
    
    def set_offset_to_current_position(self, fix_sync_pos=None):
        """ Fix the sync of the axes without moving the motors. The offset for each axis is adjust to align with the fix_sync_position."""
                
        fix_sync_pos = fix_sync_pos if fix_sync_pos is not None else self.sync.position
        curr_pos = {}
        for attr in self._real_attrs:
            curr_pos[attr] = getattr(self, attr).position
            #    pos = sync * scale + offset
            self.offsets[attr] = curr_pos[attr] - fix_sync_pos * self.scales[attr]
        
        self._setup_offsets()

class EpicsSignalROStop(EpicsSignalRO):
    def stop(self):
        return
    
class EpicsSignalStop(EpicsSignal):
    def stop(self):
        return
    
class ScanCallback(CallbackBase):
    def __init__(self, filepath = '/cds/home/opr/tmoopr/saved_data/'):
        self._filepath = filepath 
        if not os.path.exists(self._filepath):
            os.makedirs(self._filepath)
        
    def start(self, doc):
        standard_keys = ['uid', 'time', 'versions', 'scan_id', 'plan_type', 'plan_name', 'detectors', 'motors', 'num_points', 'num_intervals', 'plan_args', 'hints', 'plan_pattern', 'plan_pattern_module', 'plan_pattern_args', 'filename']
        self._start_doc = doc
        self.data = OrderedDict()
        self.msg = doc.get('purpose', None)
        self.data['detectors']=doc['detectors']
        self.data['motors']=doc.get('motors', [''])
        for key in doc.keys():
            if not (key in standard_keys):
                self.data[key]=[doc[key]]
        
        if 'filename' in doc:
            self._filename = doc['filename']
        else:
            self._filename = '_'.join(doc['detectors'] )
            
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
        for key in doc['timestamps'].keys():
            self.data[(key + ' timestamp')].append(doc['timestamps'][key])       
        return

    def stop(self, doc):
        self._end_doc = doc
        self._filename += '_%s.h5' % (datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        with h5py.File(self._filepath + self._filename, 'w') as hf_dat:
            for key in self.data.keys():
                hf_dat[key] = self.data[key]        
        if self.msg is None: self.msg=''
        self.msg += '\nSaved data to %s' % (self._filepath + self._filename)        
        self.msg += '\nscan exit status: %s' % doc['exit_status']
        logger.info(self.msg) 
        elog.post(msg=self.msg, tags='scan')
        return 
