import sys
import os
from tqdm import tqdm
import json
import ujson
import logging
import time
from time import sleep, time
from itertools import combinations, combinations_with_replacement, permutations
import ast
import math
from datetime import datetime

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import pyvisa as visa
from pyvisa.highlevel import ResourceManager

import TimeTagger
# import thorlabs_apt as thorlabs
import pylablib.devices.Thorlabs.kinesis as kin
# https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control

from ThorlabsPM100 import ThorlabsPM100
import pickle
if not hasattr(sys, 'argv'):
    sys.argv = ['']

def save_json(data, filepath):
    """Save data using json"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def save_pickle(data, filepath):
    """Save data using pickle - fast for large files"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


##### POWER ANGLE SCAN 

"""
# install instructions if there is a package conflict
pip install pyvisa
pip install pyusb

installling libusb : 
    run: conda update -n base -c defaults conda
    run: conda install -c anaconda git
        if Clobber Error run: conda clean --all
    run: git clone https://github.com/Theano/Theano.git
    
    run: conda install vcpkg
    download Visual Studio
    run: vcpkg install libusb
        if error: in triplet x86-windows: Unable to find a valid Visual Studio instance // Could not locate a complete Visual Studio instance
        Visual Studio Installer->installed version->modify-> Language package->add English package, can resolve this problem
        ... -> modify ->  ->install a Windows SDK 
        try select and install "Visual C++ tools for CMake" in the Visual Studio Installer <- This worked!
    
pip install libusb
pip install ThorlabsPM100

"""

# redundant?
# if not hasattr(sys, 'argv'):
#     sys.argv = ['']



def move_rot_stage(stage, target):
    # move rotation stage to a target angel
    stage.move_to(target) 
    while stage.is_in_motion:
        sleep(2)
    return 


def read_power(power_meter):
    # read power, convert to mW and round to 3 digits after comma
    power_read = np.round(float(power_meter.query('measure:power?')) * 1E3, 3)
    sleep(1)
    return power_read


def do_power_scan(rot_stage, power_meter, target_avg_powers, rot_start_pos, accuracy):
    # find the positions of the rotation stage at which the target average 
    # powers are obtained. These positions are returned as a list.
    # Cases not covered: Powers are out of range, bad starting positions
    print("****************")
    print("Starting power scan ...")
    print("Target Powers [mW]: ", target_avg_powers)
    
    rot_final_positions = []
    target_avg_powers = target_avg_powers
    rot_start_pos = rot_start_pos
    acc = accuracy  # mW
    
    rot_stage_current_pos = rot_start_pos
    move_rot_stage(rot_stage, rot_stage_current_pos)

    for target_pow in tqdm(target_avg_powers, desc=f'Progress '):
        print("\nCurrent target : ", target_pow, " mW ")
        delta = np.round(read_power(power_meter) - target_pow, 4)
        
        while np.abs(delta) > acc:
            # repeat until difference to target is less or equal to accuracy 
            print(r"Delta to target : ", delta , " mW\r")
                            
            step = np.abs(delta)*0.2
            # set next position
            if delta < 0:
                rot_stage_current_pos += step
            else:
                rot_stage_current_pos -= step
                
            # move stage  
            move_rot_stage(rot_stage, rot_stage_current_pos)
            delta = np.round(read_power(power_meter) - target_pow, 4)
        
        rot_final_positions.append(np.round(rot_stage_current_pos, 4))
        
    return rot_final_positions

def check_powers(ROT_STAGE, PM100D, TARGET_POWERS, rot_final_positions, ACC):

    print("****************")
    print("\nChecking ...")
    
    for power, angle in zip(TARGET_POWERS, rot_final_positions):
        move_rot_stage(ROT_STAGE, angle)
        delta = np.abs(np.round(read_power(PM100D) - power, 4))
        print("Target : ", power, " mW", " Actual :", read_power(PM100D),"mW",
              " Delta : ", delta)
        #if delta > ACC:
        #    print("Delta to large. Refreshing value.")
        #    updated_angle = do_power_scan(ROT_STAGE, PM100D, [power], [angle], ACC)
        #    rot_final_positions[:] = [updated_angle if x==angle else x for x in rot_final_positions]
        #    check_powers(ROT_STAGE, PM100D, TARGET_POWERS, rot_final_positions, ACC)
            
        
    print("****************")
    print("Target Powers [mW]: ", repr(TARGET_POWERS))
    print("Final stage positions [deg]: \n", rot_final_positions)


def run_power_angle_scan(target_powers, logger):

    logger.info("Starting Power - Angle search script.")

    rm = visa.ResourceManager()
    # print(rm.list_resources())
    # This might change depending on the USB bus used. Check in print statement 
    # from above
    identificant = "USB0::0x1313::0x8078::P0041950::INSTR"
    PM100D  = rm.open_resource(identificant)

    # some basic functions
    # print(PM100D.read)
    #print(PM100D.query('measure:power?'))
    #print(PM100D.query('power:dc:unit?'))
    #print(PM100D.query('sense:corr:wav?'), "nm")

    serial = thorlabs.list_available_devices()[0][1]
    ROT_STAGE = thorlabs.Motor(serial)

    #  print(f"Rotation stage `init: {[True if ROT_STAGE else False]}")
   #  print(f"Power Meter init: {[True if PM100D else False]}")

    TARGET_POWERS = np.arange(25, 100, 5) # mW
    ROT_START_POS = 45 # degree
    ACC = 0.3 # mW
    rot_final_positions = do_power_scan(ROT_STAGE, PM100D, TARGET_POWERS, ROT_START_POS, ACC)
    # print("****************")
    # print("Target Powers [mW]: ", repr(TARGET_POWERS))
    # print("Final stage positions [deg]: \n", rot_final_positions)
    check_powers(ROT_STAGE, PM100D, TARGET_POWERS, rot_final_positions, ACC)

    # TODO : Improve the checkup. Give already checked positions and optimize from there so its faster
    # Flip so that measuerments start with highest avg power
    powers = np.flip(TARGET_POWERS)
    angles = np.flip(rot_final_positions)
        
    
    return rot_final_positions, TARGET_POWERS

def get_time_date():
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_hms = now.strftime("%H-%M-%S")
    return time_hms, date

def init_save_dir(save_dir):
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    return


def init_tagger(logger):
    
    
    """
    Function to warm-up the Time Tagger until temperatures are stable
    or wait for the specified duration (Time Tagger 20).
    """
        # Searching for available Time Taggers
    serials = TimeTagger.scanTimeTagger()
    logger.info('Found {} connected Time Tagger(s)'.format(len(serials)))
    
    # Connect to Time Tagger and collect device information.
    # Please specify serial number explicitly in case more than one device is connected
    # and not already taken care of by a synchronizer.
    try:
        tagger = TimeTagger.createTimeTagger(serial='')
    except:
        logger.error("No Time Tagger found. Is it occupied by another program?")
        return
    serial = tagger.getSerial()
    model = tagger.getModel()
    lic_info = tagger.getDeviceLicense()
    edition = "TBD" #lic_info['edition']
    
    if model == 'Time Tagger Ultra' and edition == 'HighRes':
        resolution_modes = ['Standard', 'HighResA', 'HighResB', 'HighResC']
    else:
        resolution_modes = ['Standard']
    tt_info = {"Serial": serial, "Model": model, "Edition": edition, "Resolution modes": ', '.join(resolution_modes)}
    # logger.info('Time Tagger info : %s', tt_info)
    
    if model not in ['Time Tagger 20', 'Time Tagger Ultra', 'Time Tagger X']:
        raise ValueError('Currently "%s" is not supported by this example script', model)
    
    # logger.warning("No Time Tagger Warmup!")
    # Removing warmup procedue
    if False:
        sleep(0.5)
        # logger.info('Warming up the Time Tagger to get accurate jitter measurement.')
    
        
        all_channels = tagger.getChannelList(TimeTagger.ChannelEdge.Rising)
        # tagger.setTestSignal(all_channels, True)
        # Create load on all channels to warm up the Time Tagger
        countrate = TimeTagger.Countrate(tagger, all_channels)
        duration=int(30e12)
        if tagger.getModel() == "Time Tagger 20":
            warmup_time = duration/1e12
            # logger.info("Warming up ... ", end=' ')
            while warmup_time > 0:
                logger.info(f'{warmup_time:0.0f}', end=' ')
                sleep(2)
                warmup_time -= 2
            # logger.info("Done.")
        else:
            warmed_up = False
            cnt = 0
            while not warmed_up:
                pcb_temperatures = []
                fpga_temperatures = []
                # Measure 5 times, then compare if temperature has been stable. If not, repeat.
                for i in range(5):
                    sleep(1)
                    cnt = cnt + 1
                    sensor_data = tagger.getSensorData()
                    pcb = sensor_data[0]['FPGA board']['Board Temp #1']
                    fpga = sensor_data[0]['FPGA board']['FPGA Temp']
                    pcb_temperatures.append(pcb)
                    fpga_temperatures.append(fpga)
                    logger.info("t: {:3d} s, Board temperature: {:2.1f} °C, FPGA temperature: {:2.1f} °C".format(cnt, pcb, fpga))
                if (max(pcb_temperatures) - min(pcb_temperatures)) < 1 and (max(fpga_temperatures) - min(fpga_temperatures)) < 1.5:
                    warmed_up = True
                    logger.info("The Time Tagger is warmed up.")
                else:
                    logger.info("The Time Tagger is not yet warmed up.")
    
    logger.info("Time Tagger initialized.")
    
    return tagger

def init_thorlabs(logger):
    try:
        print("Available devices :", thorlabs.list_available_devices())
        serial = thorlabs.list_available_devices()[0][1]
        print(serial)
        motor = thorlabs.Motor(serial)
        # motor.go_home()???
        logger.info(f"Initialized thorlabs stage {serial}")
        return motor
    except:
        logger.warning(f"No thorlabs stage found")
        return None


def synchronized_correlation_measurement(tagger, logger, **kwargs):
    """
    For the jitter measurement, we use correlation measurements of the periodic built-in test signal.
    The function initializes multiple correlation measurements between channels.
    To have a simultaneous start, we make use of the Synchronized Measurement class.
    Returns correlation histogram data for each channel combination.
    """
    logger.info("Executing synchronized correlation measuerment.")
    correlation_measurements = {}
    
    # Use helper class to synchronize measurements
    synchronized = TimeTagger.SynchronizedMeasurements(tagger)
    
    if "test" in kwargs["experimental"]["savepath"]:
        logger.warning("TEST SIGNAL")
        [tagger.setTestSignal(channel, True) for channel in kwargs["general"]["timetagging"]["channels"]]
    
    # logger.info("Setting threshold to 0.5V")
    for ch in kwargs["general"]["timetagging"]["channels"]:
        tagger.setTriggerLevel(ch, 0.5)
    
    # Initialize Conditional Filter
    if kwargs["general"]["timetagging"]["filter"] and kwargs["general"]["timetagging"]["trigger"]:
        logger.info("Conditional filter enabled.")
        tagger.setConditionalFilter(trigger = kwargs["general"]["timetagging"]["trigger"],
                            filtered = kwargs["filter"],
                            hardwareDelayCompensation = True)        
        
    if kwargs["general"]["timetagging"]["delays"]:
        logger.info("Delays on channels enabled.")
        for channel, delay in zip(kwargs["general"]["timetagging"]["channels"],
                                   kwargs["general"]["timetagging"]["delays"]):
            tagger.setInputDelay(channel, delay)
            actual_delay = tagger.getInputDelay(channel)
            logger.info(f"Delay for channel {channel}: requested={delay} ps, actual={actual_delay} ps.")
    # Print TI fighters
    print(3*"Running ... ")
    # SELECT EXPERIMENT TYPE BY FLAGS
    if kwargs["general"]["experiment_type"].lower() == "g2":
        logger.info("Initializing g2 measurement.")
        with synchronized as sync_meas:
            # All possible channel combination are used for Start-Stop configuration.
            for start_channel in kwargs["general"]["timetagging"]["channels"]:
                for stop_channel in kwargs["general"]["timetagging"]["channels"]:
              
                    # Correlation Measuerments
                    corr_channels = (start_channel, stop_channel)
                    # Initiate the measurements and register them for the synchronized measurement
                    corr = TimeTagger.Correlation(sync_meas.getTagger(), start_channel, stop_channel,
                                                  kwargs["general"]["timetagging"]["binwidth_ps"],
                                                  kwargs["general"]["timetagging"]["num_bins"])
                    correlation_measurements[corr_channels] = corr
                    # Countrate Measurement
                    ctrate = TimeTagger.Countrate(sync_meas.getTagger(),
                                                  kwargs["general"]["timetagging"]["channels"])
            # Start all measurements simultaneously
            duration_min = kwargs["experimental"]["duration"] * 1E-12 / 60
            logger.info(f"Measurement starting for {duration_min:.3f}min / {duration_min/60:.3f}h")
            sync_meas.startFor(kwargs["experimental"]["duration"])
            sync_meas.waitUntilFinished()
            
            logger.info("Finished synchronized correlation measuerment.")
           
            # Collect correlation results
            corr_histograms = {}
            for corr_channels, corr in correlation_measurements.items():
                corr_histograms[str(corr_channels)] = (corr.getIndex().tolist(), corr.getData().tolist())
                # Collect count rate results
            countrate_results = {}
            for ctr_channel, index in zip(kwargs["general"]["timetagging"]["channels"], np.arange(0, len(kwargs["general"]["timetagging"]["channels"]))):
                countrate_results[str(ctr_channel)] = (ctrate.getData()[index].tolist(), ctrate.getCountsTotal()[index].tolist())
                print("Countrate results: ", countrate_results[str(ctr_channel)])
            return corr_histograms, countrate_results
    
    elif kwargs["general"]["experiment_type"].lower() == "g2_heralded_v2":
        logger.info("Initializing g2 and heralded g2 measuerurement.")

        with synchronized as sync_meas:
                       
            # Countrate Measurement
            logger.info("Initializing single countrate acquisition.")
            ctrate = TimeTagger.Countrate(sync_meas.getTagger(), kwargs['general']['timetagging']["channels"])
            
            # All possible channel combination are used for Start-Stop configuration.
            # init g2 measuerment
            logger.info("Initializing g2 acquisition.")
            g2_channel_combinations = [comb for comb in combinations(kwargs['general']['timetagging']["channels"], 2)]
            
            for comb_ch in g2_channel_combinations:
                    # Correlation Measuerments
                    start_channel, stop_channel = comb_ch[0], comb_ch[1]
                    corr_channels = (start_channel, stop_channel)
                    # Initiate the measurements and register them for the synchronized measurement
                    corr = TimeTagger.Correlation(sync_meas.getTagger(), start_channel, stop_channel,
                                                  kwargs['general']['timetagging']["binwidth_ps"], 
                                                  kwargs['general']['timetagging']["num_bins"])
                    correlation_measurements[corr_channels] = corr
            for channel in kwargs['general']['timetagging']["channels"]:
                    # Autocorrelation of detector with itself
                    start_channel, stop_channel = channel, channel
                    corr_channels = (start_channel, stop_channel)
                    # Initiate the measurements and register them for the synchronized measurement
                    corr = TimeTagger.Correlation(sync_meas.getTagger(), start_channel, stop_channel,
                                                  kwargs['general']['timetagging']["binwidth_ps"],
                                                  kwargs['general']['timetagging']["num_bins"])
                    correlation_measurements[corr_channels] = corr
         
            # G3 measuerment
            # Gate detector will be the first channel given in the kwargs
            logger.info("Initializing g3 acquisition.")
            
            corr_channels = [[1, 3, 4], [1, 5, 6], [3, 1, 2], [3, 5, 6], [6, 1, 2], [6, 3, 4], [1, 3, 5], [2, 4, 6]]
            # corr_channels = (kwargs["channels"][5], kwargs["channels"][0], kwargs["channels"][1])
            for corr_channel in corr_channels:
                corr2d = TimeTagger.Histogram2D(sync_meas.getTagger(),
                                                         start_channel=corr_channel[0], 
                                                         stop_channel_1=corr_channel[1],
                                                         stop_channel_2=corr_channel[2],
                                                         binwidth_1=kwargs['general']['timetagging']["binwidth_ps"], 
                                                         binwidth_2=kwargs['general']['timetagging']["binwidth_ps"], 
                                                         n_bins_1=500,
                                                         n_bins_2=500)
                
                correlation_measurements[tuple(corr_channel)] = corr2d 

                           
           
            # Start all measurements simultaneously
            duration_min = kwargs["experimental"]["duration"] * 1E-12 / 60
            logger.info(f"Measurement starting for {duration_min:.3f}min / {duration_min/60:.3f}h")
            sync_meas.startFor(kwargs["experimental"]["duration"])
            sync_meas.waitUntilFinished()
            
            logger.info("Finished synchronized correlation measuerment.")
           
            # Collect correlation results
            corr_histograms = {}
            for corr_channels, corr in correlation_measurements.items():
                logger.info(f"Collecting correlation data for {corr_channels}")
                corr_histograms[str(corr_channels)] = (corr.getIndex().tolist(), corr.getData().tolist())
                # Collect count rate results
            countrate_results = {}
            for ctr_channel, index in zip(kwargs["general"]["timetagging"]["channels"], np.arange(0, len(kwargs["general"]["timetagging"]["channels"]))):
                logger.info(f"Collecting countrate data for {ctr_channel}")
                countrate_results[str(ctr_channel)] = (ctrate.getData()[index].tolist(), ctrate.getCountsTotal()[index].tolist())
                    
            return corr_histograms, countrate_results
        
    elif kwargs["general"]["experiment_type"].lower() == "g2_heralded_virtual":
        logger.info("Initializing measurement with virtual mode channels for ALL heralding permutations.")
        
        # Parameter Setup 
        channels = kwargs["general"]["timetagging"]['channels']
        mode_on_channel = kwargs["general"]["timetagging"]['mode_on_channel']
        coincidence_window = kwargs["general"]["timetagging"]['coincidence_window']
    
        # Internal Mode Generation 
        modes = {}
        for channel, mode_string in zip(channels, mode_on_channel):
            mode_name = mode_string[:-1]
            if mode_name not in modes:
                modes[mode_name] = []
            modes[mode_name].append(channel)
        logger.info(f"Automatically generated modes from input: {modes}")
    
        # Results Dictionary - reorganized structure
        results = {
            "counts_physical": {},
            "counts_virtual": {},
            "countrates_physical": {},
            "countrates_virtual": {},
            "coincidences_twofold_physical": {},
            "coincidences_twofold_virtual": {},
            "correlations_virtual": {},
            "heralded_threefold": {}
        }
        
        # Virtual channel object storage
        _vc_store = []
    
        # Create combined virtual channels
        logger.info("Setting up virtual channels...")
        mode_vcs = {}
        for mode_name, physical_channels in modes.items():
            combiner = TimeTagger.Combiner(tagger, physical_channels)
            _vc_store.append(combiner) 
            mode_vcs[mode_name] = combiner.getChannel()
        
        mode_names = list(mode_vcs.keys())
        all_mode_channels = list(mode_vcs.values())
        
        # Create heralding channels by coincidence
        if len(mode_names) < 3: 
            herald_permutations = []
        else: 
            herald_permutations = list(permutations(mode_names, 3))
    
        heralded_s1_vcs = {}
        for h_mode, s1_mode, s2_mode in herald_permutations:
            perm_key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
            h_ch, s1_ch = mode_vcs[h_mode], mode_vcs[s1_mode]
            coincidence_vc = TimeTagger.Coincidence(
                tagger, [h_ch, s1_ch], 
                coincidenceWindow=coincidence_window, 
                timestamp=TimeTagger.CoincidenceTimestamp_ListedFirst
            )
            _vc_store.append(coincidence_vc)
            heralded_s1_vcs[perm_key] = coincidence_vc
    
        # Setup measurements
        with TimeTagger.SynchronizedMeasurements(tagger) as sync_meas:
            sm_tagger = sync_meas.getTagger()
            
            # --- Physical channel measurements ---
            physical_counts_counter = TimeTagger.Countrate(sm_tagger, channels)
            
            physical_coincidence_combinations = list(combinations(channels, 2))
            physical_coincidences_generator = TimeTagger.Coincidences(
                sm_tagger, 
                coincidenceGroups=physical_coincidence_combinations,
                coincidenceWindow=coincidence_window
            )
            physical_coincidence_counter = TimeTagger.Countrate(
                sm_tagger,
                channels=physical_coincidences_generator.getChannels()
            )
    
            # --- Virtual channel measurements ---
            mode_counts_counter = TimeTagger.Countrate(sm_tagger, all_mode_channels)
            
            virtual_coincidence_combinations = list(combinations(all_mode_channels, 2))
            virtual_coincidences_generator = TimeTagger.Coincidences(
                sm_tagger, 
                coincidenceGroups=virtual_coincidence_combinations,
                coincidenceWindow=coincidence_window
            )
            virtual_coincidence_counter = TimeTagger.Countrate(
                sm_tagger,
                channels=virtual_coincidences_generator.getChannels()
            )
    
            # --- Correlations between virtual modes ---
            g2_mode_correlations = {}
            for mode1, mode2 in combinations_with_replacement(mode_names, 2):
                g2_mode_correlations[f"({mode1},{mode2})"] = TimeTagger.Correlation(
                    sm_tagger, mode_vcs[mode1], mode_vcs[mode2],
                    kwargs["general"]["timetagging"]['binwidth_ps'],
                    kwargs["general"]["timetagging"]['num_bins']
                )
    
            # --- Heralded measurements ---
            g2h_numerators = {}
            g2h_denominators = {}
            g2h_rate_counters = {}
            
            for h_mode, s1_mode, s2_mode in herald_permutations:
                perm_key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
                h_ch, s2_ch = mode_vcs[h_mode], mode_vcs[s2_mode]
                heralded_vc = heralded_s1_vcs[perm_key]
    
                # Correlation: heralded_s1 with s2
                g2h_numerators[perm_key] = TimeTagger.Correlation(
                    sm_tagger, heralded_vc.getChannel(), s2_ch,
                    kwargs["general"]["timetagging"]['binwidth_ps'],
                    kwargs["general"]["timetagging"]['num_bins']
                )
                # Correlation: herald with s2 (for normalization later)
                g2h_denominators[perm_key] = TimeTagger.Correlation(
                    sm_tagger, h_ch, s2_ch,
                    kwargs["general"]["timetagging"]['binwidth_ps'], 
                    kwargs["general"]["timetagging"]['num_bins']
                )
                # Rates for herald and heralded channels
                g2h_rate_counters[perm_key] = TimeTagger.Countrate(
                    sm_tagger, [h_ch, heralded_vc.getChannel()]
                )
            
            # --- Execute All Measurements ---
            duration_min = kwargs["experimental"]["duration"] * 1E-12 / 60
            logger.info(f"Measurement starting for {duration_min:.3f}min / {duration_min/60:.3f}h")
            
            sync_meas.startFor(int(kwargs["experimental"]["duration"]))
            sync_meas.waitUntilFinished()
            logger.info("Measurements finished.")
            
        # --- Data Collection ---
        logger.info("Collecting measurement data...")
    
        # Physical counts and rates
        for ch, count, rate in zip(channels, 
                                   physical_counts_counter.getCountsTotal(), 
                                   physical_counts_counter.getData()):
            results["counts_physical"][str(ch)] = int(count)
            results["countrates_physical"][str(ch)] = float(rate)
    
        # Virtual counts and rates
        for mode_name, count, rate in zip(mode_names, 
                                          mode_counts_counter.getCountsTotal(), 
                                          mode_counts_counter.getData()):
            results["counts_virtual"][mode_name] = int(count)
            results["countrates_virtual"][mode_name] = float(rate)
    
        # Physical twofold coincidences
        for comb, count in zip(physical_coincidence_combinations, 
                               physical_coincidence_counter.getCountsTotal()):
            key = f"({comb[0]},{comb[1]})"
            results["coincidences_twofold_physical"][key] = int(count)
    
        # Virtual twofold coincidences
        for i, comb in enumerate(virtual_coincidence_combinations):
            # Map channel numbers back to mode names
            ch_to_mode = {ch: name for name, ch in mode_vcs.items()}
            mode1, mode2 = ch_to_mode[comb[0]], ch_to_mode[comb[1]]
            key = f"({mode1},{mode2})"
            results["coincidences_twofold_virtual"][key] = int(
                virtual_coincidence_counter.getCountsTotal()[i]
            )
    
        # Correlations between virtual modes (raw data only)
        for key, corr_obj in g2_mode_correlations.items():
            results["correlations_virtual"][key] = {
                "time_bins": corr_obj.getIndex().tolist(),
                "counts": corr_obj.getData().tolist()
            }
    
        # Heralded threefold data
        for h_mode, s1_mode, s2_mode in herald_permutations:
            perm_key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
            
            h_rate, h_s1_rate = g2h_rate_counters[perm_key].getData()
            h_counts, h_s1_counts = g2h_rate_counters[perm_key].getCountsTotal()
            
            results["heralded_threefold"][perm_key] = {
                "numerator": {
                    "time_bins": g2h_numerators[perm_key].getIndex().tolist(),
                    "counts": g2h_numerators[perm_key].getData().tolist()
                },
                "denominator": {
                    "time_bins": g2h_denominators[perm_key].getIndex().tolist(),
                    "counts": g2h_denominators[perm_key].getData().tolist()
                },
                "rates": {
                    "herald": float(h_rate),
                    "heralded_s1": float(h_s1_rate)
                },
                "counts": {
                    "herald": int(h_counts),
                    "heralded_s1": int(h_s1_counts)
                }
            }
    
        return results
    
    
def setup_timetagger(tagger, logger, **kwargs):
    # Keep the serial of the already opened device
    serial = tagger.getSerial()

    # Re-open with the requested resolution mode
    resolution = getattr(
        TimeTagger.Resolution,
        kwargs["general"]["timetagging"]["tt_mode"]
    )
    # If needed in your setup, free the previous handle first:
    # TimeTagger.freeTimeTagger(tagger)
    tagger = TimeTagger.createTimeTagger(serial=serial, resolution=resolution)

    # Check available inputs for the selected mode
    if kwargs["general"]["timetagging"]["tt_mode"] == "Standard":
        available_inputs = tagger.getChannelList(TimeTagger.ChannelEdge.Rising)
    else:
        available_inputs = tagger.getChannelList(TimeTagger.ChannelEdge.HighResRising)

    # Validate requested user channels
    user_channels = kwargs["general"]["timetagging"]["channels"]
    if not set(user_channels).issubset(set(available_inputs)):
        logger.warning("All or one selected channel is not available.")
        logger.info(f"Available channels: {available_inputs} / Specified channels: {user_channels}")
        logger.warning("Stopping.")
        return None

    # Validate trigger/filter channels if used
    filter_channels = kwargs["general"]["timetagging"].get("filter", [])
    trigger_channels = kwargs["general"]["timetagging"].get("trigger", [])
    if not set(filter_channels + trigger_channels).issubset(set(available_inputs)):
        logger.warning("All or one selected trigger/filter channel is not available.")
        logger.warning("Stopping.")
        return None

    # Set dead times
    for channel, deadtime in zip(
        kwargs["general"]["timetagging"]["channels"],
        kwargs["general"]["timetagging"]["deadtime"]
    ):
        tagger.setDeadtime(channel, deadtime)
        logger.info(f"Deadtime for channel {channel}: {deadtime} ps")

    # Set input delays
    delays = kwargs["general"]["timetagging"].get("delays", [])
    if delays:
        if len(delays) != len(user_channels):
            logger.warning("Length mismatch between channels and delays.")
            logger.warning("Stopping.")
            return None

        logger.info("Delays on channels enabled.")
        for channel, delay in zip(user_channels, delays):
            tagger.setInputDelay(channel, delay)
            actual_delay = tagger.getInputDelay(channel)
            logger.info(
                f"Delay for channel {channel}: requested={delay} ps, actual={actual_delay} ps"
            )

    return tagger

def perform_experiment(tagger, logger, **kwargs):
    """
    :param parameters: dictionary containing the experimental parameters
    {type: [g2, g3]
    :return: describe what it returns
    """ 
    
    logger.info("Initializing experiment on Time Tagger.")
    if tagger:
 
            setup_timetagger(tagger, logger, **kwargs)
            # END OF TIME TAGGER SETUP
            if kwargs["experimental"]["type"].lower() == "g2_heralded_virtual":
                results = synchronized_correlation_measurement(tagger, logger, **kwargs)
            
                
                # custom data saving
                data = {"Parameters": kwargs, 
                        "data": results}
                logger.info("Starting writing data to file.")
                                
                fp = str(kwargs["experimental"]["savepath"]) + ".pkl"
                save_pickle(data, fp)
                logger.info("Saved data.")

                
            if kwargs["experimental"]["type"].lower() == "g2_heralded_v2":
                    corr_histogramm, countrate_results = synchronized_correlation_measurement(tagger, logger, **kwargs)
                
                    
                    # custom data saving
                    data = {"Parameters": kwargs, 
                            "Correlation": corr_histogramm, 
                            "Countrate": countrate_results}
                    logger.info("Starting writing data to file.")
                                    
                    fp = str(kwargs["experimental"]["savepath"]) + ".pkl"
                    save_pickle(data, fp)
                    logger.info("Saved data.")
                    
            if kwargs["experimental"]["type"].lower() == "g2":
                corr_histogramm, countrate_results = synchronized_correlation_measurement(tagger, logger, **kwargs)
                        
                            
                # custom data saving
                data = {"Parameters": kwargs, 
                        "Correlation": corr_histogramm, 
                        "Countrate": countrate_results}
                logger.info("Starting writing data to file.")
                                            
                fp = str(kwargs["experimental"]["savepath"]) + ".pkl"
                save_pickle(data, fp)
                logger.info("Saved data.")
            
            # if  kwargs["experimental"]["type"].lower() != "lqt":
            #     # CONFIGURATION AND EXECUTION OF EXPERIMENT ON TIMETAGGER
            #     corr_histogramm, countrate_results = synchronized_correlation_measurement(tagger, logger, **kwargs)
            
                
            #     # custom data saving
            #     data = {"Parameters": kwargs, 
            #             "Correlation": corr_histogramm, 
            #             "Countrate": countrate_results}
            #     logger.info("Starting writing data to file.")
                                
            #     fp = str(kwargs["experimental"]["savepath"]) + ".pkl"
            #     save_pickle(data, fp)
            #     logger.info("Saved data.")
            #     TimeTagger.freeTimeTagger(tagger)
            #     logger.info("Time Tagger freed.")
            # else:
            #     # we need some custom data saving for the linear tomography experiment
                
            #     correlation_data, coincidences_data, countrate_results = synchronized_correlation_measurement(tagger, logger, **kwargs)
                
            #     # custom data saving
            #     data = {"Parameters": kwargs, 
            #             "Correlation": correlation_data.tolist(), 
            #             "Coincidences": coincidences_data.tolist(),
            #             "Countrate": countrate_results}
                
            #     # fp = str(kwargs["savepath"]) + ".json"

            #     # with open(fp, 'w', encoding='utf-8') as f:
            #     #     json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            #     print("saving")
            #     str(kwargs["savepath"]) + ".pkl"
            #     save_pickle(data, fp)
            #     print("after saving")
    else:
            logger.error("No time tagger after initialization.")
            logger.warning("Stopping.")

    return

def plot_data(data):
    
    channels = data["Parameters"]["channels"]
    mode_on_channel = data["Parameters"]["mode_on_channel"]
    fig, (ax1) = plt.subplots(1)
    
    ax1.plot(data["Correlation"][f"({channels[0]}, {channels[1]})"][0],
             data["Correlation"][f"({channels[0]}, {channels[1]})"][1],
             alpha=0.6, label=f"{mode_on_channel[0]} / {mode_on_channel[1]}")
    ax1.set_yscale("log")
    ax1.set_xlabel('Time difference (ps)')
    ax1.set_ylabel('Counts')
    #ax1.plot(data["Correlation"]["(1, 3)"][0], data["Correlation"]["(1, 3)"][1], alpha=0.6, label='H3/H5 ' )
    ax1.set_title(f'Correlation Counts/nDuration {data["Parameters"]["duration"] / 1E12} s')
    #ax1.plot(data["Correlation"]["(3, 4)"][0], data["Correlation"]["(3, 4)"][1], alpha=0.6, label='H5/H5 ')
    ax1.legend()
            
    plt.savefig(data["Parameters"]["savepath"] + ".png", dpi=400)
    plt.close()
        

def logger(SAVE_DIR):
    """Set up logging to file and console."""
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    
    # Use a named logger instead of root logger
    logger = logging.getLogger("experiment")
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    fileHandler = logging.FileHandler("{0}/{1}.log".format(SAVE_DIR, SAVE_DIR.name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    logger.info("Script start.")
    logger.propagate = False
    
    return logger

# CHUNKING
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 03 13:13:13 2025

Example Script to run a simultaneous g2 & g3 experiment for heralding purposes

@author: DT
"""
from datetime import datetime
from experiment_utils import *
from pylablib.devices import Thorlabs
import math
import pickle

def get_duration_for_power(power_mw):
    """Map power to appropriate measurement duration"""
    for range_name, (min_p, max_p, hours) in POWER_DURATION_MAP.items():
        if min_p < power_mw <= max_p:
            return int(hours * 60 * 60 * 1E12)
    return int(1 * 60 * 60 * 1E12)


def check_coincidence_threshold(merged_data, min_counts=100):
    """
    Check if all coincidence combinations have reached the minimum count threshold.
    
    Parameters
    ----------
    merged_data : dict
        Merged data dictionary with structure {"Parameters": ..., "data": ...}
    min_counts : int
        Minimum required counts for each coincidence combination
    
    Returns
    -------
    reached : bool
        True if all combinations have >= min_counts
    status : dict
        Dictionary with counts for each combination and whether threshold is met
    """

    assert merged_data["Parameters"]["experimental"]["type"] == "g2_heralded_virtual"
    results = merged_data["data"]
    status = {
        "twofold_physical": {},
        # "twofold_virtual": {},
        # "heralded_threefold": {},
        "all_reached": True,
        "min_counts": min_counts
    }
    
    # Check physical twofold
    for key, count in results.get("coincidences_twofold_physical", {}).items():
        reached = count >= min_counts
        status["twofold_physical"][key] = {"counts": count, "reached": reached}
        if not reached:
            status["all_reached"] = False
    
    # # Check virtual twofold
    # for key, count in results.get("coincidences_twofold_virtual", {}).items():
    #     reached = count >= min_counts
    #     status["twofold_virtual"][key] = {"counts": count, "reached": reached}
    #     if not reached:
    #         status["all_reached"] = False
    
    # # Check heralded threefold (use heralded_s1 counts as metric)
    # for perm_key, data in results.get("heralded_threefold", {}).items():
    #     count = data["counts"]["heralded_s1"]
    #     reached = count >= min_counts
    #     status["heralded_threefold"][perm_key] = {"counts": count, "reached": reached}
    #     if not reached:
    #         status["all_reached"] = False
    
    return status["all_reached"], status


def merge_experiment_data(data1, data2):
    """
    Merge two experiment data dictionaries from g2_heralded_virtual experiments.
    """
    import numpy as np
    
    
    if data1["Parameters"]["experimental"]["type"] == "g2_heralded_virtual":
        assert data2["Parameters"]["experimental"]["type"] == "g2_heralded_virtual"
        duration1_ps = data1["Parameters"]["experimental"]["duration"]
        duration2_ps = data2["Parameters"]["experimental"]["duration"]
        total_duration_ps = duration1_ps + duration2_ps
        
        w1 = duration1_ps / total_duration_ps
        w2 = duration2_ps / total_duration_ps
        
        results1 = data1["data"]
        results2 = data2["data"]
        
        merged_results = {
            "counts_physical": {},
            "counts_virtual": {},
            "countrates_physical": {},
            "countrates_virtual": {},
            "coincidences_twofold_physical": {},
            "coincidences_twofold_virtual": {},
            "correlations_virtual": {},
            "heralded_threefold": {}
        }
        
        # Counts: sum
        for key in results1.get("counts_physical", {}):
            merged_results["counts_physical"][key] = (
                results1["counts_physical"][key] + results2["counts_physical"][key]
            )
        
        for key in results1.get("counts_virtual", {}):
            merged_results["counts_virtual"][key] = (
                results1["counts_virtual"][key] + results2["counts_virtual"][key]
            )
        
        # Countrates: weighted average
        for key in results1.get("countrates_physical", {}):
            merged_results["countrates_physical"][key] = (
                w1 * results1["countrates_physical"][key] + 
                w2 * results2["countrates_physical"][key]
            )
        
        for key in results1.get("countrates_virtual", {}):
            merged_results["countrates_virtual"][key] = (
                w1 * results1["countrates_virtual"][key] + 
                w2 * results2["countrates_virtual"][key]
            )
        
        # Coincidences: sum
        for key in results1.get("coincidences_twofold_physical", {}):
            merged_results["coincidences_twofold_physical"][key] = (
                results1["coincidences_twofold_physical"][key] + 
                results2["coincidences_twofold_physical"][key]
            )
        
        for key in results1.get("coincidences_twofold_virtual", {}):
            merged_results["coincidences_twofold_virtual"][key] = (
                results1["coincidences_twofold_virtual"][key] + 
                results2["coincidences_twofold_virtual"][key]
            )
        
        # Correlations: sum histograms
        for key in results1.get("correlations_virtual", {}):
            merged_results["correlations_virtual"][key] = {
                "time_bins": results1["correlations_virtual"][key]["time_bins"],
                "counts": (
                    np.array(results1["correlations_virtual"][key]["counts"]) + 
                    np.array(results2["correlations_virtual"][key]["counts"])
                ).tolist()
            }
        
        # Heralded threefold
        for perm_key in results1.get("heralded_threefold", {}):
            h1 = results1["heralded_threefold"][perm_key]
            h2 = results2["heralded_threefold"][perm_key]
            
            merged_results["heralded_threefold"][perm_key] = {
                "numerator": {
                    "time_bins": h1["numerator"]["time_bins"],
                    "counts": (
                        np.array(h1["numerator"]["counts"]) + 
                        np.array(h2["numerator"]["counts"])
                    ).tolist()
                },
                "denominator": {
                    "time_bins": h1["denominator"]["time_bins"],
                    "counts": (
                        np.array(h1["denominator"]["counts"]) + 
                        np.array(h2["denominator"]["counts"])
                    ).tolist()
                },
                "rates": {
                    "herald": w1 * h1["rates"]["herald"] + w2 * h2["rates"]["herald"],
                    "heralded_s1": w1 * h1["rates"]["heralded_s1"] + w2 * h2["rates"]["heralded_s1"]
                },
                "counts": {
                    "herald": h1["counts"]["herald"] + h2["counts"]["herald"],
                    "heralded_s1": h1["counts"]["heralded_s1"] + h2["counts"]["heralded_s1"]
                }
            }
        
        # Merge parameters
        merged_params = data1["Parameters"].copy()
        merged_params["experimental"] = data1["Parameters"]["experimental"].copy()
        merged_params["experimental"]["duration"] = total_duration_ps
        
        return {"Parameters": merged_params, "data": merged_results}
    else:
        exp_type = data1["Parameters"]["experimental"]["type"]
        raise TypeError(f"MERGING NOT IMPLEMENT FOR TYPE: {exp_type}")


def merge_multiple_experiment_data(data_list):
    """Merge multiple experiment data dictionaries sequentially."""
    if len(data_list) == 0:
        raise ValueError("Cannot merge empty list")
    if len(data_list) == 1:
        return data_list[0]
    
    merged = data_list[0]
    for data in data_list[1:]:
        merged = merge_experiment_data(merged, data)
    return merged


def print_coincidence_status(status, merged_data, logger):
    """Pretty print the coincidence status with countrates and duration."""
    
    results = merged_data["data"]
    duration_ps = merged_data["Parameters"]["experimental"]["duration"]
    duration_min = duration_ps * 1e-12 / 60
    duration_h = duration_min / 60
    
    logger.info("=" * 60)
    logger.info(f"STATUS REPORT | Duration: {duration_min:.2f} min ({duration_h:.3f} h)")
    logger.info("=" * 60)
    
    # Countrates - Physical
    logger.info("COUNTRATES (Physical):")
    for ch, rate in results.get("countrates_physical", {}).items():
        count = results.get("counts_physical", {}).get(ch, 0)
        logger.info(f"  Ch {ch}: {rate*1E-3:.1f} kHz (total: {count:,})")
    
    # Countrates - Virtual
    logger.info("COUNTRATES (Virtual):")
    for mode, rate in results.get("countrates_virtual", {}).items():
        count = results.get("counts_virtual", {}).get(mode, 0)
        logger.info(f"  {mode}: {rate*1E-3:.1f} kHz (total: {count:,})")
    
    logger.info("-" * 60)
    logger.info(f"COINCIDENCES (threshold: {status['min_counts']})")
    logger.info("-" * 60)
    
    # Physical twofold
    logger.info("Twofold (physical) (Reached Threshold):")
    for key, info in status["twofold_physical"].items():
        marker = "Y" if info["reached"] else "X"
        logger.info(f"  {key}: {info['counts']:,} {marker}")
    
    logger.info("=" * 60)
    if status["all_reached"]:
        logger.info(">>> ALL THRESHOLDS REACHED <<<")
    else:
        not_reached = sum(1 for info in status["twofold_physical"].values() if not info["reached"])
        logger.info(f">>> {not_reached} combination(s) below threshold <<<")
    logger.info("=" * 60)
