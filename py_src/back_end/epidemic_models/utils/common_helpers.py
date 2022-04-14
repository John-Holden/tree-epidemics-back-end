import math
import numpy as np
import datetime as dt
from typing import Optional, List
from py_src.params_and_config import (DomainConfig, Compartmentalised_models, GenericSimulationConfig, SaveOptions, 
                                      RuntimeSettings, PATH_TO_TEMP_STORE)


def time_print(time_seconds: int, msg: Optional[str] = 'Simulation done in: ', display: Optional[bool] = True) -> str:
    """
    Pretty formatter to display time
    """
    seconds = math.floor(time_seconds)
    hrs = math.floor(seconds / 3600)
    mns = math.floor(seconds / 60)
    secs = seconds % 60

    if seconds < 60:
        if display:
            print(f'{msg} {seconds} (s)')
    elif 60 <= seconds < 3600:
        if display:
            print(f'{msg} {mns} (mins): {secs} (s)')
    elif seconds >= 3600:
        if display:
            print(f'{msg} {hrs} (Hrs): {mns%60} (mins): {secs} (s)')

    return f'{msg} {hrs} (Hrs): {mns%60} (mins): {secs} (s)'


def get_initial_host_number(patch_size: tuple, tree_density: float) -> float:
    """
    Calculate number of susceptible hosts in a simple flat domain of size [Lx, Ly]
    """
    return patch_size[0] * patch_size[1] * tree_density


def simple_square_host_num(S: List[np.ndarray], I: List[np.ndarray], R: List[np.ndarray]) -> int:
    """
    From list of SIR fields, return the total number of hosts in the system
    """
    return len(S[0]) + len(I[0]) + len(R[0])


def get_total_host_number(S: List[np.ndarray], I: List[np.ndarray], R: List[np.ndarray],
                          domain_config: DomainConfig) -> int:
    # Find and return the total number of trees in the domain based on SIR
    if domain_config.domain_type == 'simple_square':
        return simple_square_host_num(S, I, R)

    raise NotImplementedError(f'domain type: {domain_config.domain_type}')


def get_tree_density(host_number: int, patch_size: tuple):
    # Find tree density based on host number and domain config
    return host_number / (patch_size[0] * patch_size[1])


def get_model_name(compartments: str, dispersal_model: str, sporulation_model: Optional[str] = None) -> str:
    """
    Get the model-name, based on sporulation and dispersal type
    :param compartments:
    :param dispersal_model:
    :param sporulation_model:

    :return:
    """

    if compartments not in Compartmentalised_models:
        raise NotImplementedError(f'Expected models {Compartmentalised_models}, found type {compartments}')

    name = False
    if sporulation_model is None:
        name = 'phi0'
    elif sporulation_model == 'step':
        name = 'phi1'
    elif sporulation_model == 'peaked':
        name = 'phi2'
    elif not compartments:
        raise Exception('Incorrectly defined sporulation')

    if 'power' in dispersal_model and 'law' in dispersal_model:
        return f'{compartments}-{name}-pl'
    elif dispersal_model in ['Gaussian', 'gaussian', 'ga']:
        return f'{compartments}-{name}-ga'


def logger(msg: str, extra: Optional[dict] = None):
    """
    Simple pretty logger
    """
    
    if extra:
        list_objects = []
        for key_value in extra.items():
            key, value = key_value
            fmt_extra = f'\n\t{key} - {value}'
            list_objects.append(fmt_extra)

        print(f'{msg}: {"".join(list_objects)}')
        return
    
    print(msg)


def write_simulation_params(sim_context: GenericSimulationConfig, save_options: SaveOptions, rt_settings: RuntimeSettings):
    """
    Write simulation parameters to file - pickek up later by c++ executable
    """
    known_inbuilts = ['count', 'index']
    sim_write_loc = f'{PATH_TO_TEMP_STORE}/{dt.datetime.now().strftime("%d%m%Y%H%M%S")}'
    # todo spike json parser...
    with open(sim_write_loc, mode='w') as write_file:
        for obj in sim_context.items():
            config_element = obj[0]
            
            if config_element == 'sim_name':
                write_file.writelines(f'sim name: {obj[1]}\n') 
                continue

            if config_element == 'domain_config':
                continue

        
            write_file.writelines(f'{obj[0]}\n') 
            
            
            for dir_obj in dir(obj[1]):
                if dir_obj.startswith('_') or dir_obj in known_inbuilts:
                    continue

                write_file.writelines(f'- {dir_obj}: \n') 
                print('relevant dir obj = ', dir_obj)
            

        

        
     