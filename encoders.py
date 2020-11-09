from abc import ABC, abstractclassmethod
import os
import json
import re
from typing import List, Callable
import subprocess
from abc import ABC, abstractmethod
from utils import VideoInfo, run_process
from anchor import AnchorCfg, VariantCfg

__encoders__ = {}

class ReferenceEncoder(ABC):

    @staticmethod
    def encode_variant(cfg:VariantCfg, **kwargs):
        raise NotImplementedError()
    
    @staticmethod
    def decode_variant(cfg:VariantCfg, **kwargs):
        raise NotImplementedError()


def register_encoder(cls:ReferenceEncoder):
    """
    a class decorator to register a custom encoder implementation
    the encoder class must declare an 'encoder_id' class property
    """
    if 'encoder_id' not in cls.__dict__.keys():
        raise Exception(f'{cls} does not declare an encoder_id')
    if cls.encoder_id in __encoders__:
        raise Exception(f'encoder id \"{cls.encoder_id}\" already registered by {__encoders__[cls.encoder_id]}')
    __encoders__[cls.encoder_id] = cls
    return cls

def get_encoder(id:str) -> 'ReferenceEncoder':
    """
    get a registered encoder implementation
    """
    return __encoders__[id]

def _to_cli_args(opts:dict):
    """
    convert dict to cli options for HM & VTM softwares
    """
    args = []
    for k, v in opts.items():
        if k.startswith('--'):
            args.append(f'{k}={v}')
        else:
            args += [k, str(v)]
    return args

def get_env(var:str):
    v = os.getenv(var)
    if v == None:
        raise Exception(f'environment variable not set {var}')
    return v

def reference_encoder_args(variant:VariantCfg, recon=True):
    """
    HM & VTM softwares share a common set of options
    """
    base = { "-c": f'{variant.anchor.encoder_cfg}',
        "-i": f'{variant.anchor.reference.path}',
        "-b": f'{variant.bitstream}' ,
        "--FrameRate": round(variant.anchor.reference.fps) ,
        "--FramesToBeEncoded": variant.anchor.reference.framecount ,
        "--SourceWidth": variant.anchor.reference.width ,
        "--SourceHeight": variant.anchor.reference.height ,
        "--InputBitDepth": variant.anchor.reference.bitdepth ,
        "--InputBitDepthC": variant.anchor.reference.bitdepth_chroma ,
        "--InputChromaFormat": variant.anchor.reference.chroma_subsampling }
    if recon:
        base["-o"] = f'{variant.reconstructed}'
    return [ *_to_cli_args(base), *_to_cli_args(variant.options) ]


@register_encoder
class HM(ReferenceEncoder):
    
    encoder_id = "HM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("HM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        run_process(logfile, encoder, *reference_encoder_args(v, recon), *_to_cli_args(opts))
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("HM_DECODER")
        args = _to_cli_args({ 
            "-b": v.bitstream,
            "-o": v.reconstructed,
            **opts })
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        run_process(logfile, decoder, *args)
        return logfile

@register_encoder
class VTM(ReferenceEncoder):
    encoder_id = "VTM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("VTM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        opl = v.anchor.working_dir / f'{v.basename}.opl'
        run_process(logfile, encoder, *reference_encoder_args(v, recon), '-opl', opl, *_to_cli_args(opts))
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("VTM_DECODER")
        args = _to_cli_args({ 
            "-b": v.bitstream,
            "-o": v.reconstructed,
            **opts })
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        run_process(logfile, decoder, *args)
        return logfile



@register_encoder
class JM(ReferenceEncoder):

    encoder_id = "JM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("JM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        """
        # 3.1 Encoder Syntax
        lencod [-h] [-d defenc.cfg] {[-f curenc1.cfg]...[-f curencN.cfg]}
                {[-p EncParam1=EncValue1]...[-p EncParamM=EncValueM]}
        """
        args = [ '-d', f'{v.anchor.encoder_cfg}',
            '-p', f'InputFile={v.anchor.reference.path}',
            "-p", f'OutputFile={v.bitstream}' ,
            "-p", f'FrameRate={float(v.anchor.reference.fps)}',
            "-p", f'FramesToBeEncoded={v.anchor.reference.framecount}',
            "-p", f'SourceWidth={v.anchor.reference.width}',
            "-p", f'SourceHeight={v.anchor.reference.height}',
            "-p", f'SourceBitDepthLuma={v.anchor.reference.bitdepth}',
            "-p", f'SourceBitDepthChroma={v.anchor.reference.bitdepth_chroma}',
            "-p", f'InputChromaFormat={v.anchor.reference.chroma_subsampling}' ]
        
        keys = v.options.keys()
        if "Interleaved" not in keys:
            args += ["-p", "Interleaved=0"]
        if "RGBInput" not in keys:
            args += ["-p", 'RGBInput=0']
        if "StandardRange" not in keys:
            args += ["-p", 'StandardRange=0']
        if "VideoCode" not in keys:
            args += ["-p", 'VideoCode=1']
        
        if v.anchor.reference.chroma_subsampling == "400":
            args += ["-p", f'YUVFormat=0']
        elif v.anchor.reference.chroma_subsampling == "420":
            args += ["-p", f'YUVFormat=1']
        elif v.anchor.reference.chroma_subsampling == "422":
            args += ["-p", f'YUVFormat=2']
        elif v.anchor.reference.chroma_subsampling == "444":
            args += ["-p", f'YUVFormat=3']
        
        if recon:
            args += ["-p", f'ReconFile={v.reconstructed}']

        for opt, val in v.options.items():
            if opt == 'f' or opt == '-f':
                args += ['-f', str(val)]
            else:
                args += ['-p', f'{opt}={val}']
        run_process(logfile, encoder, *args) 
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("JM_DECODER")
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        """
        ldecod [-s] [-h] [-d] [defdec.cfg]
            {[-f curenc1.cfg]...[-f curencN.cfg]}
            [-i bitstream.264] [-o output.yuv] [-r reference.yuv]
            {[-p DecParam1=DecValue1]...[-p DecParamM=DecValueM]}
            [-n] Nframes [-mpr] LValue
        """
        args = [ "-i", v.bitstream,
            "-o", v.reconstructed,
            "-r", v.anchor.reference.path ] 
        run_process(logfile, decoder, *args)
        return logfile