from abc import ABC, abstractclassmethod
import os
import re
import shlex
from pathlib import Path
from typing import List
from utils import run_process, ChromaFormat, ChromaSubsampling
from anchor import AnchorTuple, VariantData, ReconstructionMeta

__encoders__ = {}

def parse_encoding_bitdepth(cfg:Path, encoder_id:str=None):
    if encoder_id != None:
        return __encoders__[encoder_id].parse_encoding_bitdepth(cfg)
    with open(cfg, 'r') as fo:
        for line in fo:
            m = re.match(f'(InternalBitDepth\s*:\s*)(\d*)', line)
            if m:
                return int(m[2])
        return -1


class EncoderBase(ABC):

    @abstractclassmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        raise NotImplementedError()
    
    @abstractclassmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
        raise NotImplementedError()

    @abstractclassmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def decoder_log_metrics(cls) -> dict:
        return {}

    @classmethod
    def encoder_log_metrics(cls) -> dict:
        return {}

    @classmethod
    def cfg(cls, a:AnchorTuple) -> dict:
        return None

    @classmethod
    def encode_variant(cls, a:AnchorTuple, variant_id:str, variant_cli:str, dst_dir:Path=None) -> VariantData:

        if dst_dir != None:
            assert dst_dir.is_dir(), 'target directory not found'
        else:
            dst_dir = a.working_dir, 'anchor working directory not found'

        encoder = get_env(cls.encoder_bin)
        bitstream = dst_dir / f'{variant_id}.bin'
        reconstruction = dst_dir / f'{variant_id}.rec.yuv'
        logfile = dst_dir / f'{variant_id}.encoder.log'
        
        cmd = cls.get_encoder_cmd(a, variant_cli, bitstream, reconstruction)
        run_process(logfile, encoder, *cmd, dry_run=a.dry_run)

        if a.dry_run:
            return VariantData()
        return VariantData.new(a, variant_id, variant_cli, bitstream, logfile)


    @classmethod
    def decode_variant(cls, a:AnchorTuple, v:VariantData, dst_dir:Path=None, md5=True) -> ReconstructionMeta:
        if dst_dir != None:
            assert a.dry_run or dst_dir.is_dir(), f'decoder output directory not found: {dst_dir}'
        else:
            dst_dir = a.working_dir
        decoder = get_env(cls.decoder_bin)
        bitstream = a.working_dir / v.bitstream['URI']
        reconstructed = dst_dir / f'{bitstream.stem}.yuv'
        logfile = dst_dir / f'{bitstream.stem}.dec.log'
        
        cmd = cls.get_decoder_cmd(bitstream, reconstructed, a)
        run_process(logfile, decoder, *cmd, dry_run=a.dry_run)
        
        return ReconstructionMeta(cls.encoder_id, reconstructed, logfile, md5=md5)
    

def register_encoder(cls:EncoderBase):
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

def get_encoder(id:str) -> 'EncoderBase':
    """
    get a registered encoder implementation
    """
    if id in __encoders__:
        return __encoders__[id]
    try:
        m = re.search('^[a-zA-Z]*', id).group()
        if m != '':
            return __encoders__[m]
    except KeyError:
        pass
    return None
    
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



def reference_encoder_args(a:AnchorTuple, bitstream:Path, reconstruction:Path=None, extra:bool=False):
    """
    HM & VTM softwares share a common set of options
    """
    cmd = { "-c": f'{a.encoder_cfg}',
        "--InputFile": f'{a.reference.path}',
        "--BitstreamFile": str(bitstream),
        "--FrameRate": round(a.reference.frame_rate) ,
        "--FrameSkip": f'{a.start_frame -1}',
        "--FramesToBeEncoded": f'{a.frame_count}' ,
        "--SourceWidth": f'{a.reference.width}' ,
        "--SourceHeight": f'{a.reference.height}' ,
        "--InputBitDepth": f'{a.reference.bit_depth}' ,
        "--InputBitDepthC": f'{a.reference.bit_depth}' ,
        "--InputChromaFormat": f'{a.reference.chroma_subsampling.value}' }

    if not a.reference.video_full_range:
        # If 1 then clip input video to the Rec. 709 Range on loading 
        #   when Internal-BitDepth is less than MSBExtendedBitDepth.
        cmd["--ClipInputVideoToRec709Range"] = 1
    else:
        cmd["--ClipInputVideoToRec709Range"] = 0
    
    assert a.reference.hdr_master_display == None, "Table 34: Mastering display colour volume SEI message encoder paramete"
    assert a.reference.hdr_max_cll == None
    assert a.reference.hdr_max_fall == None

    if reconstruction != None:
        cmd["--ReconFile"] = str(reconstruction)
        # /* ITU-R BT.709 compliant clipping for converting say 10b to 8b */
        cmd["--ClipOutputVideoToRec709Range"] = cmd["--ClipInputVideoToRec709Range"]
    if extra:
        cmd["--PrintMSSSIM"] = 1
        cmd["--PrintHexPSNR"] = 1

    # explicit defaults ######################################

    # Extends the input video by adding MSBs of value 0. When 0, no extension
    # is applied and the InputBitDepth is used.
    # The MSBExtendedBitDepth becomes the effective file InputBitDepth for
    # subsequent processing.
    cmd["--MSBExtendedBitDepth"] = 0 

    # Specifies the bit depth used for coding. When 0, the setting defaults to the
    # value of the MSBExtendedBitDepth.
    # If the input video is a different bit depth to InternalBitDepth, it is automati-
    # cally converted [...] The effect of this option is as if the input video is externally converted
    # to the MSBExtendedBitDepth and then to the InternalBitDepth and then
    # coded with this value as InputBitDepth. The codec has no notion of different
    # bit depths.
    cmd["--InternalBitDepth"] = 0

    # When this is set true, then no colour space conversion is applied prior to
    # PSNR calculation, otherwise the inverse of InputColourSpaceConvert is ap-
    # plied.
    cmd["--InputColourSpaceConvert"] = "UNCHANGED"
    cmd["--SNRInternalColourSpace"] = False
    cmd["--OutputInternalColourSpace"] = 0
    # cmd["--ChromaFormatIDC"] = 0 # = InputChromaFormat

    return _to_cli_args(cmd)


@register_encoder
class HM(EncoderBase):
    
    encoder_id = os.getenv("HM_VERSION", "HM")
    encoder_bin = "HM_ENCODER"
    decoder_bin = "HM_DECODER"

    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        qp_args = args.replace('-qp', '-q')
        return shlex.split(qp_args)
    
    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None, extra:bool=False) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction, extra)
        args += cls.get_variant_cli(variant_cli)
        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        """
        -d,   --OutputBitDepthC        bit depth of YUV output chroma component
                                 (default: use 0 for native depth)
        --OutputColourSpaceConvert
                                 Colour space conversion to apply to input 444
                                 video. Permitted values are (empty
                                 string=UNCHANGED) UNCHANGED, YCrCbtoYCbCr or
                                 GBRtoRGB
        --ClipOutputVideoToRec709Range /* ITU-R BT.709 compliant clipping for converting say 10b to 8b */
        """
        return [
            *_to_cli_args({ "-b": f'{bitstream}', "-o": f'{reconstructed}', "-d": f'{a.reference.bit_depth}' }),
            "--ClipOutputVideoToRec709Range"
        ]


@register_encoder
class SCM(HM):

    encoder_id = os.getenv("SCM_VERSION", "SCM")
    encoder_bin = "SCM_ENCODER"
    decoder_bin = "SCM_DECODER"

    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        return super().get_variant_cli(args)

    @classmethod
    def get_encoder_cmd(cls, *args, **kwargs):
        return super().get_encoder_cmd(*args, **kwargs)

    @classmethod
    def get_decoder_cmd(cls, *args, **kwargs):
        return super().get_decoder_cmd(*args, **kwargs)



@register_encoder
class VTM(EncoderBase):
    
    encoder_id = os.getenv("VTM_VERSION", "VTM")
    encoder_bin = "VTM_ENCODER"
    decoder_bin = "VTM_DECODER"

    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        qp_args = args.replace('-qp', '-q')
        return shlex.split(qp_args)

    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None, extra:bool=False) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction, extra)
        args += cls.get_variant_cli(variant_cli)
        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        return _to_cli_args({ "-b": f'{bitstream}', "-o": f'{reconstructed}' })



@register_encoder
class JM(EncoderBase):

    encoder_id = os.getenv("JM_VERSION", "JM")
    encoder_bin = "JM_ENCODER"
    decoder_bin = "JM_DECODER"

    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        qp = args.split()[-1]
        qp_args = f'-p QPISlice={qp} -p QPPSlice={qp}'
        return shlex.split(qp_args)

    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None, extra:bool=False) -> List[str]:
        # tracefile = v.anchor.working_dir / f'{v.basename}.enc.trace.txt'
        # statsfile = v.anchor.working_dir / f'{v.basename}.enc.stats.dat'

        args = [ "-p", "DisplayEncParams=1",
            "-d", f'{a.encoder_cfg}',
            # "-p", f'TraceFile={tracefile}',
            # "-p", f'StatsFile={statsfile}',
            "-p", f'InputFile={a.reference.path}',
            "-p", f'OutputFile={bitstream}',
            "-p", f'FrameRate={float(a.reference.frame_rate)}',
            "-p", f'StartFrame={a.start_frame -1}',
            "-p", f'FramesToBeEncoded={a.frame_count}',
            "-p", f'SourceWidth={a.reference.width}',
            "-p", f'SourceHeight={a.reference.height}',
            "-p", f'OutputWidth={a.reference.width}',
            "-p", f'OutputHeight={a.reference.height}',
            "-p", f'SourceBitDepthLuma={a.reference.bit_depth}',
            "-p", f'SourceBitDepthChroma={a.reference.bit_depth}',
        ]

        if a.reference.interleaved:
            args += ["-p", "Interleaved=1"]

        if a.reference.chroma_format == ChromaFormat.RGB:
            args += ["-p", 'RGBInput=1', "-p", 'StandardRange=1'] # 1 = full range
        else:
            args += ["-p", 'RGBInput=0', "-p", 'StandardRange=0']
        
        if a.reference.chroma_subsampling == ChromaSubsampling.CS_400:
            args += ["-p", f'YUVFormat=0']
        elif a.reference.chroma_subsampling ==  ChromaSubsampling.CS_420:
            args += ["-p", f'YUVFormat=1']
        elif a.reference.chroma_subsampling ==  ChromaSubsampling.CS_422:
            args += ["-p", f'YUVFormat=2']
        elif a.reference.chroma_subsampling ==  ChromaSubsampling.CS_444:
            args += ["-p", f'YUVFormat=3']

        if reconstruction != None:
            args += ["-p", f'ReconFile={reconstruction}']

        args += cls.get_variant_cli(variant_cli)

        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        args = [ "-i", f'{bitstream}', "-o", f'{reconstructed}' ] 
        # "-r", f'{a.reference.path}' ] 
        return args
    

@register_encoder
class ETM(EncoderBase):

    encoder_id = os.getenv("ETM_VERSION", "ETM")
    encoder_bin = "ETM_ENCODER"
    decoder_bin = "ETM_DECODER"

    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        qp_args = args.replace('-qp', '-q')
        return shlex.split(qp_args)

    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None, extra:bool=False) -> List[str]:
        assert a.reference.chroma_format == ChromaFormat.YUV, 'RGB chroma format not supported'
        assert a.reference.bit_depth in [8, 10], f'invalid reference bitdepth {a.reference.bit_depth} | supported: [8,10]'
        assert not a.reference.interleaved, 'interleaved format not supported'
        cf = {
            "400": 0,
            "420": 1,
            "422": 2,
            "444": 3
        }[str(a.reference.chroma_subsampling.value)]
        args = [
            '--config', f'{a.encoder_cfg}',
            '-i', f'{a.reference.path}',
            '-o', f'{bitstream}',
            '-w', f'{a.reference.width}',
            '-h', f'{a.reference.height}',
            '-z', f'{a.reference.frame_rate}',
            '-f', f'{a.reference.frame_count}',
            # --codec_bit_depth / codec internal bitdepth (10(default), 8, 12, 14)
            '-d', f'{a.reference.bit_depth}',
            '--chroma_format', f'{cf}',
            # '--hdr_metric' requires specific  compilation flags
        ]
        if reconstruction:
            # --output_bit_depth / output bitdepth (8, 10)(default: same as input bitdpeth)
            args += ['-r', reconstruction ]
        
        # MS-SSIM_Y is printed by default, no hex PSNR

        args += cls.get_variant_cli(variant_cli)
        
        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        opl = bitstream.with_suffix('.opl')
        args = [
            "-i", f'{bitstream}',
            "-o", f'{reconstructed}',
            "--opl", f'{opl}',
            # "-f", f'{a.reference.frame_count}',
            "--output_bit_depth", f'{a.reference.bit_depth}', # defaults to 8 otherwise
            "-v", '1' # 0=quiet, 2=verbose
        ]
        return args