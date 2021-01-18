import os
import re
import json
import hashlib

from pathlib import Path
from enum import Enum
from typing import List

from utils import VideoSequence, VideoInfo

RE_WORKING_DIR = r'^{ANCHOR_DIR}'

def _preproc(param, anchor:'AnchorTuple'):
    if type(param) == str:
        return re.sub(RE_WORKING_DIR, f'{anchor.working_dir}', param)
    return param

def md5_checksum(p:Path):
    md5 = hashlib.md5()
    block_size = 128 * md5.block_size
    with open(p, 'rb') as f:
        chunk = f.read(block_size)
        while chunk:
            md5.update(chunk)
            chunk = f.read(block_size)
        return md5.hexdigest()

class VariantCfg:
    
    def __init__(self, anchor:'AnchorTuple', variant_id:str, options:dict):
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
    def encoder_log(self):
        return self.anchor.working_dir / f'{self.basename}.enc.log'

    @property
    def reconstructed(self):
        return self.anchor.working_dir /  f'{self.basename}.yuv'

    @property
    def decoder_log(self):
        return self.anchor.working_dir / f'{self.basename}.dec.log'



def resolve_path(f:str, rootdir:Path):
    p = Path(f)
    if p.is_absolute():
        return p
    r = rootdir / f
    if not r.exists():
        raise Exception(f'file not found {r}')
    return r

class AnchorTuple:
    
    def __init__(self, reference:VideoSequence, encoder_id:str, encoder_cfg:str, variants={}, description:str=None, start_frame:int=0, frame_count:int=None, codecs:str=None, dry_run:bool=False):
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._variants = { k: VariantCfg(self, k, opts) for k, opts in variants.items() }
        self._start_frame = start_frame
        self._frame_count = frame_count if frame_count != None else reference.frame_count
        self._codecs = codecs

        self.dry_run = dry_run

    @property
    def working_dir(self):
        return Path(self.encoder_cfg).resolve().parent

    @property
    def codecs(self):
        return self._codecs

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
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._frame_count / self.reference.frame_rate

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
            
            rootdir = Path(file).parent
            data = json.load(f)

            src = resolve_path(data["reference"], rootdir)
            test_sequence = VideoSequence.from_sidecar_metadata(Path(src).resolve())
            
            encoder_id = data["encoder_id"]
            encoder_cfg = resolve_path(data["encoder_cfg"], rootdir)
            
            if "variants" in data:
                variants = data["variants"]
            else:
                variants = {}
            
            # anchor may be defined as a segment of the reference sequence, otherwise the entire sequence gets used
            if "reference_segment" in data:
                segment = data["reference_segment"]
                start_frame = segment.get('start_frame', 0)
                frame_count = segment.get('frame_count', 1)
            
            description = data.get("description", None)
            return AnchorTuple(test_sequence, encoder_id, encoder_cfg, variants, description, start_frame, frame_count)
            