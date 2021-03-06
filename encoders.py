from abc import ABC, abstractclassmethod
import os
import json
import re
from typing import List, Callable
import subprocess
from abc import ABC, abstractmethod
from utils import run_process, VideoInfo, ChromaFormat, ChromaSubsampling
from anchor import VariantCfg

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
        "--InputFile": f'{variant.anchor.reference.path}',
        "--BitstreamFile": f'{variant.bitstream}',
        "--FrameRate": round(variant.anchor.reference.frame_rate) ,
        "--FrameSkip": f'{variant.anchor.start_frame}',
        "--FramesToBeEncoded": f'{variant.anchor.frame_count}' ,
        "--SourceWidth": f'{variant.anchor.reference.width}' ,
        "--SourceHeight": f'{variant.anchor.reference.height}' ,
        "--InputBitDepth": f'{variant.anchor.reference.bit_depth}' ,
        "--InputBitDepthC": f'{variant.anchor.reference.bit_depth}' ,
        "--InputChromaFormat": f'{variant.anchor.reference.chroma_subsampling.value}' }
    
    if recon:
        base["--ReconFile"] = f'{variant.reconstructed}'
    return [ *_to_cli_args(base), *_to_cli_args(variant.options) ]


@register_encoder
class HM(ReferenceEncoder):
    
    encoder_id = "HM"

    @classmethod
    def get_config_dict(cls, v:VariantCfg) -> dict:
        pass
    
    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("HM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        run_process(logfile, encoder, *reference_encoder_args(v, recon), *_to_cli_args(opts), dry_run=v.anchor.dry_run)
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("HM_DECODER")
        args = _to_cli_args({ 
            "-b": f'{v.bitstream}',
            "-o": f'{v.reconstructed}',
            **opts })
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        run_process(logfile, decoder, *args, dry_run=v.anchor.dry_run)
        return logfile



@register_encoder
class VTM(ReferenceEncoder):
    encoder_id = "VTM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("VTM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        # opl = v.anchor.working_dir / f'{v.basename}.opl'
        run_process(logfile, encoder, *reference_encoder_args(v, recon), *_to_cli_args(opts), dry_run=v.anchor.dry_run)
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("VTM_DECODER")
        args = _to_cli_args({ 
            "-b": f'{v.bitstream}',
            "-o": f'{v.reconstructed}',
            **opts })
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        run_process(logfile, decoder, *args, dry_run=v.anchor.dry_run)
        return logfile



@register_encoder
class JM(ReferenceEncoder):

    encoder_id = "JM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("JM_ENCODER")
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'
        tracefile = v.anchor.working_dir / f'{v.basename}.enc.trace.txt'
        statsfile = v.anchor.working_dir / f'{v.basename}.enc.stats.dat'
        args = [ "-p", "DisplayEncParams=1",
            '-d', f'{v.anchor.encoder_cfg}',
            "-p", f'TraceFile={tracefile}',
            "-p", f'StatsFile={statsfile}',
            "-p", f'InputFile={v.anchor.reference.path}',
            "-p", f'OutputFile={v.bitstream}',
            "-p", f'FrameRate={float(v.anchor.reference.frame_rate)}',
            "-p", f'StartFrame={v.anchor.start_frame}',
            "-p", f'FramesToBeEncoded={v.anchor.frame_count}',
            "-p", f'SourceWidth={v.anchor.reference.width}',
            "-p", f'SourceHeight={v.anchor.reference.height}',
            "-p", f'OutputWidth={v.anchor.reference.width}',
            "-p", f'OutputHeight={v.anchor.reference.height}',
            "-p", f'SourceBitDepthLuma={v.anchor.reference.bit_depth}',
            "-p", f'SourceBitDepthChroma={v.anchor.reference.bit_depth}',
        ]

        if v.anchor.reference.interleaved:
            args += ["-p", "Interleaved=1"]

        if v.anchor.reference.chroma_format == ChromaFormat.RGB:
            args += ["-p", 'RGBInput=1', "-p", 'StandardRange=1'] # 1 = full range
        else:
            args += ["-p", 'RGBInput=0', "-p", 'StandardRange=0']
        
        if v.anchor.reference.chroma_subsampling == ChromaSubsampling.CS_400:
            args += ["-p", f'YUVFormat=0']
        elif v.anchor.reference.chroma_subsampling ==  ChromaSubsampling.CS_420:
            args += ["-p", f'YUVFormat=1']
        elif v.anchor.reference.chroma_subsampling ==  ChromaSubsampling.CS_422:
            args += ["-p", f'YUVFormat=2']
        elif v.anchor.reference.chroma_subsampling ==  ChromaSubsampling.CS_444:
            args += ["-p", f'YUVFormat=3']
        
        if recon:
            args += ["-p", f'ReconFile={v.reconstructed}']

        for opt, val in v.options.items():
            if opt == 'f' or opt == '-f':
                args += ['-f', str(val)]
            else:
                args += ['-p', f'{opt}={val}']
        run_process(logfile, encoder, *args, dry_run=v.anchor.dry_run) 
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
        args = [ "-i", f'{v.bitstream}',
            "-o", f'{v.reconstructed}',
            "-r", f'{v.anchor.reference.path}' ] 
        run_process(logfile, decoder, *args, dry_run=v.anchor.dry_run)
        return logfile


@register_encoder
class ETM(ReferenceEncoder):

    encoder_id = "ETM"

    @staticmethod
    def encode_variant(v:VariantCfg, recon=True, **opts):
        encoder = get_env("ETM_ENCODER")
        # opl = v.anchor.working_dir / f'{v.basename}.opl'
        logfile = v.anchor.working_dir / f'{v.basename}.enc.log'

        assert v.anchor.reference.chroma_format == ChromaFormat.YUV, 'RGB chroma format not supported'
        assert v.anchor.reference.bit_depth in [8, 10], f'invalid reference bitdepth {v.anchor.reference.bit_depth} | supported: [8,10]'
        assert not v.anchor.reference.interleaved, 'interleaved format not supported'
        cf = {
            "400": 0,
            "420": 1,
            "422": 2,
            "444": 3
        }[str(v.anchor.reference.chroma_subsampling.value)]
        args = [
            '--config', f'{v.anchor.encoder_cfg}',
            '-i', f'{v.anchor.reference.path}',
            '-o', f'{v.bitstream}',
            '-w', f'{v.anchor.reference.width}',
            '-h', f'{v.anchor.reference.height}',
            '-z', f'{v.anchor.reference.frame_rate}',
            '-f', f'{v.anchor.reference.frame_count}',
            # --codec_bit_depth / codec internal bitdepth (10(default), 8, 12, 14)
            '-d', f'{v.anchor.reference.bit_depth}',
            '--chroma_format', f'{cf}',
            # '--hdr_metric' requires specific  compilation flags
        ]
        if recon:
            # --output_bit_depth / output bitdepth (8, 10)(default: same as input bitdpeth)
            args += ['-r', v.reconstructed ]
        run_process(logfile, encoder, *args, dry_run=v.anchor.dry_run)
        return logfile

    @staticmethod
    def decode_variant(v:VariantCfg, **opts):
        decoder = get_env("ETM_DECODER")
        opl = v.anchor.working_dir / f'{v.basename}.opl'
        args = [
            "-i", f'{v.bitstream}',
            "-o", f'{v.reconstructed}',
            "--opl", f'{opl}',
            "-f", f'{v.anchor.reference.frame_count}',
            "--output_bit_depth", f'{v.anchor.reference.bit_depth}', # defaults to 8 otherwise
            "-v", '1' # 0=quiet, 2=verbose
        ]
        logfile = v.anchor.working_dir / f'{v.basename}.dec.log'
        run_process(logfile, decoder, *args)
        return logfile