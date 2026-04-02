# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:27:33 2026

@author: Aaron
"""


import tomllib # import tomli
from pathlib import Path
from .models import Common, Confocal, LLS, Detailed_Balance, Config, ImageDir, Experiment

def load_config(path: str = None, microscope_type: str = 'confocal') -> Config:
    path = path or Path(__file__).parent / "config.toml"

    with open(path, "rb") as f:
        data = tomllib.load(f) #tomli.load(f)

    ## set variable microscope params
    im_params = (
            Confocal(**data["im_params"]["confocal"])
            if microscope_type == "confocal"
            else LLS(**data["im_params"]["lls"])
        )
    db_params = (
            Detailed_Balance(**data["db_params"]["confocal"])
            if microscope_type == "confocal"
            else Detailed_Balance(**data["db_params"]["lls"])
        )
    

    return Config(
        common=Common(**data["common"]),
        microscope = microscope_type,
        im_params=im_params,
        db_params=db_params,
        experiment=Experiment(
            galv=ImageDir(**data["data"]["galv"]),
            ck666=ImageDir(**data["data"]["ck666"]),
            pnb=ImageDir(**data["data"]["pnb"]),
            lls=ImageDir(**data["data"]["lls"]),
        ),
    )