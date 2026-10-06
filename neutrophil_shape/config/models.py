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
    pilr_method: str
    nisos: list
    all_symmetrical_pcs: dict  # per-alignment lists of symmetrical PCs, keyed by alignment name
    all_pc_flips: dict  # per-alignment lists of PC indices to flip the order of, keyed by alignment name
    savedir: str = field(init=False, default = None)
    align_method: str = field(init=False, default = None)
    normal_method: str = field(init=False, default = None)
    symmetrical_pcs: list = field(init=False, default = None)
    pc_flips: list = field(init=False, default = None)
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
    all_symmetrical_origins: dict
    cycle_thresh: dict
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
        self.common.pc_flips = self.common.all_pc_flips[value]
        ### variably set which PCs are symmetrical for this alignment
        self.common.symmetrical_pcs = self.common.all_symmetrical_pcs[value]
        ### create the combos of symmetrical PCs
        sym_pcs = self.common.symmetrical_pcs
        pc_combos_sym = [
            tuple(-c if c in sym_pcs else c for c in combo)
            for combo in self.common.pc_combos
        ]
        self.common.pc_combos_sym = pc_combos_sym
        self.common.pc_combos_sym_only = list(
            set(pc_combos_sym) - set(self.common.pc_combos)
            )[::-1]
        ### change the savedir based on the alignment
        self.common.savedir = self.common.basedir.joinpath(
            'data',
            value + '_' + self.microscope,
            )
        self.common.savedir.mkdir(parents=True, exist_ok=True)
        ### set the origins from all_origins based on alignment
        self.db_params.origins = self.db_params.all_origins[value]
        self.db_params.origins_sym = self.db_params.all_symmetrical_origins[value]
    