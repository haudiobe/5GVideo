import os
import re
import json
import hashlib

from pathlib import Path
from enum import Enum
from typing import List, Dict, Generator, Tuple

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

class ReconstructionMeta:
    def __init__(self, decoder_id:str, reconstructed:Path, decoder_log:Path, md5=True):
        self.decoder_id = decoder_id
        self.reconstructed = reconstructed
        if md5:
            self.reconstructed_md5 = md5_checksum(self.reconstructed)
        else:
            self.reconstructed_md5 = None
        self.decoder_log = decoder_log

class BitstreamNotFound(Exception):
    pass

class MetricsNotFound(Exception):
    pass

class VariantData:

    def __init__(self, generation:dict=None, bitstream:dict=None, reconstruction:dict=None, metrics:dict=None, verification:dict=None, contact:dict=None, copyright:str=''):
        self._generation = generation
        self._bitstream = bitstream
        self._contact = contact
        self._metrics = metrics
        self._reconstruction = reconstruction
        self._verification = verification
        self._copyright = copyright


    @property
    def variant_id(self) -> str:
        return self._bitstream["key"]

    @property
    def variant_cli(self) -> str:
        return self._generation["variant"]
    

    #######################################

    @property
    def generation(self) -> dict:
        return self._generation

    @generation.setter
    def generation(self, b:dict):
        if b == None:
            self._generation = None
        keys = [ "config-file", "encoder", "key", "log-file", "sequence", "variant" ]
        self._generation = { k: b[k] for k in keys }

    #######################################

    @property
    def bitstream(self) -> dict:
        return self._bitstream

    @bitstream.setter
    def bitstream(self, b:dict):
        if b == None:
            self._bitstream = None
        keys = [ "URI", "key", "md5", "size" ]
        self._bitstream = { k: b[k] for k in keys }

    #######################################

    @property
    def reconstruction(self) -> dict:
        return self._reconstruction

    @reconstruction.setter
    def reconstruction(self, b:dict):
        if b == None:
            self._reconstruction = None
        keys = [ "decoder", "log-file", "md5" ]
        self._reconstruction = { k: b[k] for k in keys }

    #######################################

    @property
    def metrics(self) -> dict:
        return self._metrics

    @metrics.setter
    def metrics(self, b:dict):
        if b == None:
            self._metrics = None
        keys = [ "Bitrate", "BitrateLog", "DecodeTime", "EncodeTime", "MS_SSIM", "UPSNR", "VMAF", "VPSNR", "YPSNR" ]
        self._metrics = { k: b[k] for k in keys }

    #######################################

    @property
    def verification(self) -> dict:
        return self._verification

    @verification.setter
    def verification(self, b:dict):
        if b == None:
            self._verification = None
        keys = [ "Reports" ] #, "cross-verification-status" ]
        self._verification = { k: b[k] for k in keys }

    #######################################

    @property
    def contact(self) -> dict:
        return self._contact

    @contact.setter
    def contact(self, b:dict):
        if b == None:
            self._contact = None
        keys = [ "Company", "email" ]
        self._contact = { k: b[k] for k in keys }

    #######################################

    @property
    def copyright(self) -> str:
        return self._copyright

    @copyright.setter
    def copyright(self, c:str):
        self._copyright = c

    #######################################

    def locate_bitstream(self, anchor_dir:Path, md5_check=True) -> Path:
        assert self.bitstream != None, 'bitstream definition not found'
        assert anchor_dir.is_dir(), 'invalid anchor directory'
        fp = anchor_dir / Path(self._bitstream['URI'])
        assert fp.exists(), f'bitstream file does not exist: {fp}'
        if md5_check:
            assert self._bitstream['md5'] == md5_checksum(fp), f'md5 missmatch: {fp}'
        return fp

    def has_metric_set(self, *metrics) -> bool:
        assert (self._reconstruction != None) and (self._reconstruction['md5'] != None), 'invalid reconstruction'
        if type(self._metrics) != dict:
            return False
        for m in metrics:
            if not m in self._metrics:
                return False
        return True

    def dumps(self):
        data = {
            "Bitstream": self._bitstream,
            "Generation": self._generation,
            "Reconstruction": self._reconstruction,
            "Metrics": self._metrics,
            "Verification": self._verification,
            "copyRight": self._copyright,
            "Contact": self._contact
        }
        return json.dumps(data, indent=4)

    def save_as(self, fp:Path):
        data = self.dumps()
        with open(fp, 'w') as fo:
            fo.write(data)

    @classmethod
    def load(cls, fp:Path) -> 'VariantData':
        assert fp.exists()
        with open(fp, 'r') as fo:
            data = json.load(fo)
            generation = data.get("Generation", None)
            bitstream = data.get("Bitstream", None)
            reconstruction = data.get("Reconstruction", None)
            metrics = data.get("Metrics", None)
            verification = data.get("Verification", None)
            contact = data.get("contact", None)
            copyright = data.get("copyRight", None)
            return VariantData(generation, bitstream, reconstruction, metrics, verification, contact, copyright)

    @classmethod
    def new(cls, a:'AnchorTuple', variant_id:str, variant_cli:str, bitstream_fp:Path, encoder_log:Path) -> 'VariantData':
        assert bitstream_fp.exists(), f'{bitstream_fp} not found'
        generation = {
            "key": variant_id,
            "sequence": a.reference.path.name,
            "encoder": a.encoder_id,
            "config-file": a.encoder_cfg,
            "variant": variant_cli,
            "log-file": encoder_log.name
        }
        bitstream = {
            "key": variant_id,
            "URI": str(bitstream_fp),
            "md5": md5_checksum(bitstream_fp),
            "size": bitstream_fp.stat().st_size
        }
        contact = {
            'Company': a.reference.contact['Company'],
            'e-mail': a.reference.contact['e-mail']
        }
        copyright = a.reference.copyright
        return VariantData(generation, bitstream, None, None, None, contact, copyright)


def resolve_path(f:str, rootdir:Path):
    p = Path(f)
    if p.is_absolute():
        return p
    r = rootdir / f
    if not r.exists():
        raise Exception(f'file not found {r}')
    return r


class AnchorTuple:
    
    def __init__(self, anchor_dir:Path, reference:VideoSequence, encoder_id:str, encoder_cfg:str, variants:str, anchor_key:str, description:str=None, start_frame:int=0, frame_count:int=None, dry_run:bool=False):
        assert anchor_dir and anchor_dir.is_dir(), f'[{anchor_key}] - invalid working directory : {anchor_dir}'
        self._working_dir = anchor_dir
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._anchor_key = anchor_key
        self._start_frame = start_frame
        self._frame_count = frame_count if frame_count != None else reference.frame_count

        assertion = "expecting 'variants' to be a list of integer QP values, parsed from a json string"
        try:
            data = json.loads(variants)
        except BaseException:
            raise ValueError(assertion)
        assert type(data) == list, assertion
        for v in data:
            assert type(v) == int, assertion
        self._variants = data

        self.dry_run = dry_run

    @property
    def working_dir(self):
        return self._working_dir

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
    def description(self) -> str:
        return self._description

    @property
    def basename(self):
        return self.encoder_cfg.stem

    def iter_variants(self) -> Generator[Tuple[str, list], None, None]:
        """
        yields (variant_id, variant_encoder_args)
        """
        for qp in self._variants:
            yield self._anchor_key.replace('<QP>', str(qp)), (qp,)
    
    @property
    def anchor_key(self):
        return self._anchor_key
