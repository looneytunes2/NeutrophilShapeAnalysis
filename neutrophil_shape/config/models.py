# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:58:16 2026

@author: Aaron
"""

# from ctypes import alignment
from dataclasses import dataclass, field
from pathlib import Path
import itertools
# from typing import Union ## for old python env


@dataclass
class ImageDir:
    serverdir: str
    localdir: str
    dates: list
    def __post_init__(self):
        self.serverdir = Path(self.serverdir)
        self.localdir = Path(self.localdir)
        self.date_dirs = [self.serverdir.joinpath(date) for date in self.dates]

@dataclass
class Common:
    smooth_factor: int
    sigma: float
    l_order: int
    npcs: int
    pcflips: list = field(init=False, default = None)
    pilr_method: str
    nisos: list
    savedir: str = field(init=False, default = None)
    align_method: str = field(init=False, default = None)
    normal_method: str = field(init=False, default = None)
    # Derived attributes
    def __post_init__(self):
        self.basedir = Path(__file__).parents[2]
        self.pc_combos = list(itertools.combinations(range(1,1+self.npcs), 2))
        
        
@dataclass
class Confocal:
    xyres: float
    zstep: float
    time_interval: float
    xy_buffer: int
    z_buffer: int
    stackshape: list
    whatseg: str
    
@dataclass
class LLS:
    xyres: float
    zstep: float
    time_interval: float
    decon: bool
    orig_size: bool
    xy_buffer: int
    z_buffer: int
    hilo: bool   

@dataclass
class Detailed_Balance:
    nbins: int
    ntrans: int
    bsiter: int
    ttot: int
    all_origins: dict
    origins: list = field(init=False, default=None)

@dataclass
class Experiment:
    galv: ImageDir
    ck666: ImageDir
    pnb: ImageDir
    lls: ImageDir

@dataclass
class Config():
    common: Common
    microscope: str
    im_params: Confocal | LLS ####  Union[Confocal, LLS] 
    db_params: Detailed_Balance
    experiment: Experiment
    alignment: type = field(init=False, default = None)

    _alignment_registry = [
        'shape',
        'trajectory_shape',
        'trajectory',
    ]

    @property
    def _alignment(self):
        return self.alignment
    
    @_alignment.setter
    def _alignment(self, value:str):
        if value not in self._alignment_registry:
            raise ValueError(f"Invalid alignment: {value}. Must be one of {self._alignment_registry}.")
        self.alignment = value
        ### variably set the INDICIES of the PCs to flip the order of
        ### need to set indicies so that components of actual pca class
        ### can be flipped too
        self.common.pc_flips = {
            'shape': [],#0,1,2,4,6],
            'trajectory_shape': [1,2,7],#0,1,2,3,6],
            'trajectory': [1,3,7],
        }[value]
        ### variably set the alignment methods based on the overall alignment
        self.common.align_method = {
            'shape':'long_axis',
            'trajectory_shape':'trajectory',
            'trajectory':'trajectory',
            }[value]
        self.common.normal_method = {
            'shape':'width',
            'trajectory_shape':'width',
            'trajectory':'planar',
            }[value]
        ### change the savedir based on the alignment
        self.common.savedir = self.common.basedir.joinpath(
            'data',
            value + '_' + self.microscope,
            )
        self.common.savedir.mkdir(parents=True, exist_ok=True)
        ### set the origins from all_origins based on alignment
        self.db_params.origins = self.db_params.all_origins[value]
    
    # Derived attributes
    def __post_init__(self):
        self.pc_combos = Path(__file__).parents[2]