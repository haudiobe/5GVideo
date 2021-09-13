from abc import ABC, abstractclassmethod
import os
import re
import shlex
from pathlib import Path
from typing import List
from utils import run_process, ChromaFormat, ChromaSubsampling, VideoSequence
from anchor import AnchorTuple, VariantData, ReconstructionMeta


__encoders__ = {}


def register_encoder(cls:'EncoderBase'):
    """a class decorator to register a custom encoder implementation
    the encoder class must declare an 'encoder_id' class property
    """
    if 'encoder_id' not in cls.__dict__.keys():
        raise Exception(f'{cls} does not declare an encoder_id')
    if cls.encoder_id in __encoders__:
        raise Exception(f'encoder id \"{cls.encoder_id}\" already registered by {__encoders__[cls.encoder_id]}')
    __encoders__[cls.encoder_id] = cls
    return cls

def get_encoder(id:str) -> 'EncoderBase':
    """get an encoder implementation decorated with @register_encoder
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
    

class EncoderBase(ABC):
    """the base class for encoders to extend
    """

    encoder_id:str = None
    encoder_bin:str = None
    decoder_bin:str = None

    @abstractclassmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        """Parse the variant cli from streams.csv into command line options.
        """
        raise NotImplementedError()
    
    @abstractclassmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
        """Produce the command line to encode the specified anchor's variant. Subclasses must supply its own implementation.
            Producing a reconstruction file while encoding is not enforced, 
                and when implemented must generate the appropriate json metadata (anchor.ReconstructionMeta) to support the metrics computation workflow.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        """Produce the command line to decoder the specified anchor's variant. Subclasses must supply its own implementation.
        """
        raise NotImplementedError()

    @classmethod
    def decoder_log_metrics(cls, log:Path) -> dict:
        """parse decoder logs / stat files, and return a metrics dict
        """
        print(f'/!\ decoder {cls} does not implement a custom log parsing')
        return {}

    @classmethod
    def encoder_log_metrics(cls, log:Path) -> dict:
        """parse encoder logs / stats, and return a metrics dict
        """
        print(f'/!\ custom encoder does not implement log parsing')
        return {}

    @classmethod
    def encode_variant(cls, a:AnchorTuple, variant_id:str, variant_cli:str, dst_dir:Path=None) -> VariantData:
        """
        encode the variant and return VariantData
        """
        if dst_dir != None:
            assert dst_dir.is_dir(), 'target directory not found'
        else:
            dst_dir = a.working_dir #, 'anchor working directory not found'
        if not a.working_dir.exists():
            a.working_dir.mkdir(parents=True)
        encoder = get_env(cls.encoder_bin)
        bitstream = dst_dir / f'{variant_id}.bin'
        reconstruction = dst_dir / f'{variant_id}.yuv'
        logfile = dst_dir / f'{variant_id}.encoder.log'
        
        cmd = cls.get_encoder_cmd(a, variant_cli, bitstream, reconstruction)
        run_process(logfile, encoder, *cmd, dry_run=a.dry_run)

        if a.dry_run:
            return VariantData()

        dist = VideoSequence(reconstruction, **a.reference.properties)
        dist.start_frame = 1
        dist.frame_count = a.reference.frame_count
        coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
        dist.bit_depth = coded_bit_depth
        dist.dump(dst_dir / f'{variant_id}.yuv.json')

        rec = ReconstructionMeta(
            a.encoder_id,
            reconstruction,
            None,
            md5=True
        )
        return VariantData.new(a, variant_id, variant_cli, logfile, bitstream, rec)


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

        dist = VideoSequence(reconstructed, **a.reference.properties)
        dist.start_frame = 1
        dist.frame_count = a.reference.frame_count
        coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
        dist.bit_depth = coded_bit_depth
        if not a.dry_run:
            dist.dump(dst_dir / f'{v.variant_id}.yuv.json')

        return ReconstructionMeta(cls.encoder_id, reconstructed, logfile, md5=md5)
    
def parse_encoding_bitdepth(cfg:Path, encoder_id:str=None):
    if encoder_id != None:
        return __encoders__[encoder_id].parse_encoding_bitdepth(cfg)
    with open(cfg, 'r') as fo:
        for line in fo:
            m = re.match(f'(InternalBitDepth\s*:\s*)(\d*)', line)
            if m:
                return int(m[2])
        return -1

def parse_variant_qp(variant_cli:str) -> int:
    return int(shlex.split(variant_cli)[-1])
    

def _to_cli_args(opts:dict):
    """convert dict to cli options for HM & VTM softwares
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



def reference_encoder_args(a:AnchorTuple, bitstream:Path, reconstruction:Path=None):
    """HM & VTM softwares share a common set of options
    """
    cmd = { 
        "-c": f'{a.encoder_cfg}',
        "--InputFile": f'{a.reference.path}',
        "--BitstreamFile": str(bitstream),
        "--FrameRate": round(a.reference.frame_rate) ,
        "--FrameSkip": f'{a.start_frame -1}',
        "--FramesToBeEncoded": f'{a.frame_count}' ,
        "--SourceWidth": f'{a.reference.width}' ,
        "--SourceHeight": f'{a.reference.height}' ,
        "--InputBitDepth": f'{a.reference.bit_depth}' ,
        "--InputBitDepthC": f'{a.reference.bit_depth}' ,
        "--InputChromaFormat": f'{a.reference.chroma_subsampling.value}',
        "--ChromaSampleLocTypeTopField": f'{a.reference.chroma_sample_loc_type}',
        "--ChromaSampleLocTypeBottomField": f'{a.reference.chroma_sample_loc_type}'
    }
    if a.reference.video_full_range:
        # If 1 then clip input video to the Rec. 709 Range on loading 
        #   when Internal-BitDepth is less than MSBExtendedBitDepth.
        cmd["--ClipInputVideoToRec709Range"] = 0
    else:
        cmd["--ClipInputVideoToRec709Range"] = 1
    
    assert a.reference.hdr_master_display == None, "Table 34: Mastering display colour volume SEI message encoder paramete"
    assert a.reference.hdr_max_cll == None
    assert a.reference.hdr_max_fall == None

    if reconstruction != None:
        cmd["--ReconFile"] = str(reconstruction)
        # /* ITU-R BT.709 compliant clipping for converting say 10b to 8b */
        # cmd["--ClipOutputVideoToRec709Range"] = cmd["--ClipInputVideoToRec709Range"]
    
    cmd["--PrintMSSSIM"] = 1
    cmd["--PrintHexPSNR"] = 1

    return _to_cli_args(cmd)


@register_encoder
class HM(EncoderBase):

    encoder_id = os.getenv("HM_VERSION", "HM")
    encoder_bin = "HM_ENCODER"
    decoder_bin = "HM_DECODER"
    
    @classmethod
    def get_variant_cli(cls, args:str) -> List[str]:
        qp = parse_variant_qp(args)
        return ['-q', str(qp)]

    
    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction)
        args += cls.get_variant_cli(variant_cli)
        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream:Path, reconstructed:Path, a:AnchorTuple) -> List[str]:
        """
        -d,   --OutputBitDepthC        bit depth of YUV output chroma component (default: use 0 for native depth)
        --OutputColourSpaceConvert
                                 Colour space conversion to apply to input 444
                                 video. Permitted values are (empty
                                 string=UNCHANGED) UNCHANGED, YCrCbtoYCbCr or
                                 GBRtoRGB
        --ClipOutputVideoToRec709Range /* ITU-R BT.709 compliant clipping for converting say 10b to 8b */
        """
        print(f'{a.reference.bit_depth}bit::{a.reference.path}')
        return [
            *_to_cli_args({ "-b": f'{bitstream}', "-o": f'{reconstructed}', "-d": "0" }),
                "--ClipOutputVideoToRec709Range"
        ]
    
    @classmethod
    def encoder_log_metrics(cls, logp:Path) -> dict:
        summary = False
        keys = None
        values = None
        with open(logp, 'r') as logf:
            for l in logf.readlines():
                if summary:
                    if keys:
                        if not values:
                            values = l.split()[2:] # '1', 'a'
                        # break
                    else:
                        keys = l.split()[3:] # 'Total', 'Frames', '|'
                else:
                    summary = l.startswith('SUMMARY')
                if values:
                    if l.startswith(' Total Time:'):
                        keys.append('EncodeTime')
                        values.append(l.split()[-2])
        assert summary and keys and values
        metrics = {}
        for i, k in enumerate(keys):
            if k == 'Bitrate':
                k = 'BitrateLog'
            elif k.endswith('PSNR'):
                if k.startswith('YUV'):
                    continue
                k = k.replace('-', '')
            elif k.endswith('Y-MS-SSIM'):
                k = 'MS_SSIM'
            elif k != 'EncodeTime':
                continue
            metrics[k] = float(values[i])
        return metrics




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
        qp = parse_variant_qp(args)
        return ['-q', str(qp)]


    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction)
        args += cls.get_variant_cli(variant_cli)
        # --LMCSSignalType         Input signal type: 0:SDR, 1:HDR-PQ, 2:HDR-HLG
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
        qp_args = f'-p QPISlice={qp} -p QPPSlice={qp}'
        return shlex.split(qp_args)

    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
        tracefile = v.anchor.working_dir / f'{v.basename}.enc.trace.txt'
        statsfile = v.anchor.working_dir / f'{v.basename}.enc.stats.dat'
        args = [ "-p", "DisplayEncParams=1",
            "-d", f'{a.encoder_cfg}',
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
            "-p", f'TraceFile={tracefile}',
            "-p", f'StatsFile={statsfile}'
        ]
        
        assert a.reference.chroma_loc_type == 2, 'chroma_loc_type != 2 not implemented'

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
        qp = parse_variant_qp(args)
        return ['-q', str(qp)]


    @classmethod
    def get_encoder_cmd(cls, a:AnchorTuple, variant_cli:str, bitstream:Path, reconstruction:Path=None) -> List[str]:
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
        
        assert a.reference.chroma_loc_type == 2, 'chroma_loc_type != 2 not implemented'

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