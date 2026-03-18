# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:58:16 2026

@author: Aaron
"""

from dataclasses import dataclass
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
    alignment: str
    smooth_factor: int
    sigma: float
    l_order: int
    pilr_method: str
    nisos: list

    # Derived attributes
    def __post_init__(self):
        self.basedir = Path(__file__).parents[1]
        self.savedir = self.basedir.joinpath(
            'data',
            self.alignment)
        self.align_method = {
            'shape':'long_axis',
            'trajectory_shape':'trajectory',
            'trajectory':'trajectory',
            }[self.alignment]
        self.normal_method = {
            'shape':'width',
            'trajectory_shape':'width',
            'trajectory':'planar',
            }[self.alignment]
        
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
class Config:
    # galv: ImageDir
    # ck666: ImageDir
    # pnb: ImageDir
    common: Common
    confocal: Confocal
    lls: LLS
    detailed_balance: Detailed_Balance