    
#%% Adaptable Stop for only for Heralded_Virtual
from experiment_utils import *
from pylablib.devices import Thorlabs
import thorlabs_rotation_stage_v2 as trs





if __name__ == "__main__":

    # SET GENERAL EXPERIMENT PARAMETERS HERE
    rep_rate_hz = 21*1E6
    wavelength_nm = 2100 
    MATERIAL = "CdTe110"
    TYPE = "g2_heralded_virtual"
    REPEATS = 2

    # Chunking settings
    CHUNK_DURATION_MINUTES = 2  # [min] - duration of each chunk
    AUTOSAVE_ENABLED = True
    SINGLE_DURATION_MINUTES = 60
    
    
    # Threshold settings
    CHECK_THRESHOLD_EVERY_N_CHUNKS = 1  # Check every N chunks
    MIN_COINCIDENCE_COUNTS = 100000    # Minimum counts required
    STOP_WHEN_THRESHOLD_REACHED = True  # Stop early if threshold met
    MAX_CHUNKS = 100                  # Safety limit (max total duration = CHUNK_DURATION * MAX_CHUNKS)
    
    BINS = int(50000)
    BINWIDTH = int(100)
    COINCIDENCE_WINDOW = int(25000)
    
    TT_MODE = "Standard"
    CHANNELS = [1, 2, 3, 4, 5, 6]
    MODE_ON_CHANNEL = ["H3T", "H3R", "H4T", "H4R", "H5T", "H5R"]
    DEAD_TIME_CHANNEL = [0, 0, 0, 0, 0, 0]
    DELAYS = [0, -1000, 20000, 16500, 17800, 17300]
    # DELAYS = [0, -1000, 20000, 165000, 17800, 200300]
    # DELAYS = [0,0,0,0,17800, 0]
    
    ID_HWP = "27264707"

    trs.connect_rotation_stages(ID_HWP)
    
    
    # Scan parameters

    angles = np.flip(np.linspace(50, 70, 35)) 
    powers = np.flip(np.linspace(5, 89, 35))
    

    BASE_DIR = Path(r"C:\Users\QUANTUM\OneDrive - ENSTA\!Working files\HBT\HBT Data\Heralding")
    

    time_hms, date = get_time_date()
    SAVE_DIR = BASE_DIR / f"{date}_{time_hms}_{TYPE}"
    
    init_save_dir(SAVE_DIR)
  
    logger = logger(SAVE_DIR)
    
    general_params = {
        "date": date,
        "time": time_hms,
        "experiment_type": TYPE,
        "save_dir": str(SAVE_DIR),
        "material": MATERIAL,
        "laser": {"rep_rate_hz": rep_rate_hz, "wavelength_nm": wavelength_nm},
        "timetagging": {
            "tt_mode": TT_MODE,
            "binwidth_ps": BINWIDTH,
            "num_bins": BINS,
            "channels": CHANNELS,
            "mode_on_channel": MODE_ON_CHANNEL,
            "deadtime": DEAD_TIME_CHANNEL,
            "delays": DELAYS,
            "coincidence_window": COINCIDENCE_WINDOW,
            "trigger": [], "filter": []
        },
        "custom": {}
    }
    
    save_json(general_params, SAVE_DIR / "general_parameters.json")
    logger.info("=" * 20 + "GENERAL SETTINGS" + "=" * 20)
    logger.info(general_params)

    tagger = init_tagger(logger)

    if tagger:
        for num_repeat in range(REPEATS):
            for current_angle, current_power in zip(angles, powers):
                current_time = time() / 60 / 60
            
                trs.rotate_stage(current_angle,ID_HWP)
                logger.info(f"Stage Position: {current_angle}")
                if AUTOSAVE_ENABLED:
                    chunk_duration_ps = int(CHUNK_DURATION_MINUTES * 60 * 1E12)
                    max_duration_min = CHUNK_DURATION_MINUTES * MAX_CHUNKS
                    
                    logger.info(f"Starting chunked acquisition (max {MAX_CHUNKS} chunks / {max_duration_min:.1f} min)")
                    logger.info(f"Chunk duration: {CHUNK_DURATION_MINUTES} min")
                    logger.info(f"Threshold: {MIN_COINCIDENCE_COUNTS} counts, checking every {CHECK_THRESHOLD_EVERY_N_CHUNKS} chunks")
                    
                    merged_data = None
                    unmerged_chunks = []
                    
                    # Single merged file path - gets overwritten each time
                    merged_dir = SAVE_DIR / "MERGED"
                    init_save_dir(merged_dir)

                    merged_fp = merged_dir / f"{time_hms}_{date}_{current_power}mW_num{num_repeat}_MERGED.pkl"
                
                    for i in range(MAX_CHUNKS):
                        tagger = init_tagger(logger)
                        
                        FILE_NAME = f"{time_hms}_{date}_{current_power}mW_num{num_repeat}_chunk{i}"
                        experiment_params = {
                            "type": TYPE,
                            "duration": chunk_duration_ps,
                            "savepath": str(SAVE_DIR / FILE_NAME),
                            "laser_power": current_power,
                            "rotation_stage": current_angle
                        }
                        
                        parameters = {"general": general_params, "experimental": experiment_params}
                        
                        init_save_dir(SAVE_DIR)
     
                        elapsed_min = (i + 1) * CHUNK_DURATION_MINUTES
                        logger.info(f"Chunk {i+1}/{MAX_CHUNKS} | Elapsed: {elapsed_min:.1f} min")
                        
                        # Run experiment
                        perform_experiment(tagger, logger, **parameters)
                        
                        # Load chunk data
                        chunk_fp = str(SAVE_DIR / FILE_NAME) + ".pkl"
                        with open(chunk_fp, 'rb') as f:
                            chunk_data = pickle.load(f)
                        unmerged_chunks.append(chunk_data)
                        
                        # Merge every N chunks to free memory
                        if (i + 1) % CHECK_THRESHOLD_EVERY_N_CHUNKS == 0:
                            logger.info(f"Merging {len(unmerged_chunks)} chunks into running total...")
                            
                            if merged_data is None:
                                merged_data = merge_multiple_experiment_data(unmerged_chunks)
                            else:
                                merged_data = merge_multiple_experiment_data([merged_data] + unmerged_chunks)
                            
                            # Free memory
                            unmerged_chunks = []
                            
                            # Threshold check
                            reached, status = check_coincidence_threshold(
                                merged_data, 
                                min_counts=MIN_COINCIDENCE_COUNTS
                            )
                            print_coincidence_status(status, merged_data, logger)
                            
                            # Save/overwrite merged file
                            with open(merged_fp, 'wb') as f:
                                pickle.dump(merged_data, f)
                            logger.info(f"Updated merged data: {merged_fp}")
                            
                            if reached and STOP_WHEN_THRESHOLD_REACHED:
                                logger.info(">>> THRESHOLD REACHED - STOPPING EARLY <<<")
                                break
                    
                    # Handle any remaining unmerged chunks
                    if unmerged_chunks:
                        logger.info(f"Merging {len(unmerged_chunks)} remaining chunks...")
                        if merged_data is None:
                            merged_data = merge_multiple_experiment_data(unmerged_chunks)
                        else:
                            merged_data = merge_multiple_experiment_data([merged_data] + unmerged_chunks)
                        
                        # Final save
                        with open(merged_fp, 'wb') as f:
                            pickle.dump(merged_data, f)
                    
                    logger.info(f"Final merged data saved to: {merged_fp}")
                    total_duration_min = merged_data["Parameters"]["experimental"]["duration"] * 1e-12 / 60
                    logger.info(f"Total merged duration: {total_duration_min:.2f} min")
                    
                else:
                    # Single acquisition, no chunking
                   
                    single_duration_ps = int(SINGLE_DURATION_MINUTES * 60 * 1E12)
                    
                    tagger = init_tagger(logger)
                    
                    FILE_NAME = f"{time_hms}_{date}_{current_power}mW_num{num_repeat}"
                    experiment_params = {
                        "type": TYPE,
                        "duration": single_duration_ps,
                        "savepath": str(SAVE_DIR / FILE_NAME),
                        "laser_power": current_power,
                        "rotation_stage": current_angle
                    }
                    
                    parameters = {"general": general_params, "experimental": experiment_params}
                    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Single acquisition | Duration: {SINGLE_DURATION_MINUTES:.1f} min")
                    perform_experiment(tagger, logger, **parameters)
                
                passed_time = (time() / 60 / 60 - current_time) * 60
                logger.info(f"Measurement completed in {passed_time:.2f} min")
                    
    else:
        logger.info(f"Motor init: {motor}")
        logger.info(f"Tagger init: {tagger}")
        logger.warning("Unexpected stop.")
    trs.disconnect_all()            
    logger.info("Successful finish.")
    del logger