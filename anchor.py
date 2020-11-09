import os
import re
import json

from pathlib import Path
from enum import Enum
from typing import List

from utils import VideoSequence, VideoInfo

RE_WORKING_DIR = r'^{ANCHOR_DIR}'

def _preproc(param, anchor:'AnchorCfg'):
    if type(param) == str:
        return re.sub(RE_WORKING_DIR, f'{anchor.working_dir}', param)
    return param

class VariantCfg:
    
    def __init__(self, anchor:'AnchorCfg', variant_id:str, options:dict):
        self.variant_id = variant_id
        self.anchor = anchor
        self.options = { k: _preproc(v, anchor) for k,v in options.items() }

    @property
    def basename(self):
        return f'{self.anchor.basename}.{self.variant_id}'

    @property
    def bitstream(self):
        return self.anchor.working_dir / f'{self.basename}.bit'

    @property
    def reconstructed(self):
        return self.anchor.working_dir /  f'{self.basename}.yuv'


class AnchorCfg:
    
    def __init__(self, reference:VideoSequence, encoder_id:str, encoder_cfg:str, variants:dict={}, description:str=None):
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._variants = { k: VariantCfg(self, k, opts) for k, opts in variants.items() }

    @property
    def working_dir(self):
        return Path(self.encoder_cfg).resolve().parent

    @property
    def encoder_id(self):
        return self._encoder_id

    @property
    def encoder_cfg(self) -> Path:
        return Path(self._encoder_cfg).resolve()

    @property
    def reference(self) -> VideoSequence:
        return self._reference

    @property
    def variants(self) -> dict:
        return self._variants.values()

    @property
    def description(self) -> str:
        return self._variants

    @property
    def basename(self):
        return self.encoder_cfg.stem
        
    @staticmethod
    def load(file:str):
        with open(file, 'r') as f:
            
            data = json.load(f)
        
            src = data["test_sequence"]
            test_sequence = VideoSequence.with_sidecar_metadata(Path(src).resolve())

            encoder_id = data["encoder_id"]
            encoder_cfg = data["encoder_cfg"]
            
            if "variants" in data:
                variants = data["variants"]
            else:
                variants = {}
            description = data.get("description", None)
            return AnchorCfg(test_sequence, encoder_id, encoder_cfg, variants, description)
            