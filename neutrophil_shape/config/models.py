# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:58:16 2026

@author: Aaron
"""

from ctypes import alignment
from dataclasses import dataclass, field
from pathlib import Path

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
    pilr_method: str
    nisos: list
    savedir: str = field(init=False, default = None)
    align_method: str = field(init=False, default = None)
    normal_method: str = field(init=False, default = None)
    # Derived attributes
    def __post_init__(self):
        self.basedir = Path(__file__).parents[1]
        
        
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
    ntrans: int
    bsiter: int

@dataclass
class Experiment:
    galv: ImageDir
    ck666: ImageDir
    pnb: ImageDir

@dataclass
class Config():
    common: Common
    confocal: Confocal
    lls: LLS
    detailed_balance: Detailed_Balance
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
        self.common.align_method = {
            'shape':'long_axis',
            'trajectory_shape':'trajectory',
            'trajectory':'trajectory',
            }[self.alignment]
        self.common.normal_method = {
            'shape':'width',
            'trajectory_shape':'width',
            'trajectory':'planar',
            }[self.alignment]
        self.common.savedir = self.common.basedir.joinpath(
            'data',
            self.alignment)
        