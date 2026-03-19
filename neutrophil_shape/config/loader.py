# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:27:33 2026

@author: Aaron
"""


import tomllib # import tomli
from pathlib import Path
from .models import Common, Confocal, LLS, Detailed_Balance, Config, ImageDir, Experiment


def load_config(path: str = None) -> Config:
    path = path or Path(__file__).parent / "config.toml"

    with open(path, "rb") as f:
        data = tomllib.load(f) #tomli.load(f)

    return Config(
        common=Common(**data["common"]),
        confocal=Confocal(**data["confocal"]),
        lls=LLS(**data["lls"]),
        detailed_balance=Detailed_Balance(**data["detailed_balance"]),
        experiment=Experiment(
            galv=ImageDir(**data["galv"]),
            ck666=ImageDir(**data["ck666"]),
            pnb=ImageDir(**data["pnb"]),
        ),
    )