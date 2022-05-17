from builtins import Exception
from ast import ExceptHandler
from abc import ABC, abstractclassmethod
import os
import re
from pathlib import Path
from typing import Any, List, Dict
from constants import Metric
from utils import run_process
from sequences import ChromaFormat, ChromaSubsampling, VideoSequence, as_8bit_sequence, as_10bit_sequence, as_exr2020_sequence
from anchor import AnchorTuple, VariantData, ReconstructionMeta

import logging as logger

__encoders__:Dict[str, 'EncoderBase'] = {}

def register_encoder(cls: 'EncoderBase') -> Any:
    """a class decorator to register a custom encoder implementation
    the encoder class must declare an 'encoder_id' class property
    """
    if 'encoder_id' not in cls.__dict__.keys():
        raise Exception(f'{cls} does not declare an encoder_id')
    if cls.encoder_id in __encoders__:
        raise Exception(f'encoder id \"{cls.encoder_id}\" already registered by {__encoders__[cls.encoder_id]}')
    __encoders__[cls.encoder_id] = cls
    return cls


def get_encoder(id: str) -> 'EncoderBase':
    """get an encoder implementation decorated with @register_encoder
    """
    if id in __encoders__:
        return __encoders__[id]
    try:
        # try matching when id is a specific version
        m = re.search('^[a-zA-Z]*', id).group()
        if m != '':
            return __encoders__[m]
    except KeyError:
        pass
    return None


class EncoderBase(ABC):
    """the base class for encoders to extend
    """
    
    encoder_id: str = None
    encoder_bin: str = None
    decoder_bin: str = None
    sei_removal_app: str = None

    @abstractclassmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        """Parse the variant cli from streams.csv into command line options.
        """
        raise NotImplementedError()
    
    @abstractclassmethod
    def get_encoder_cmd(cls, a: AnchorTuple, variant_qp: str, bitstream: Path, reconstruction: Path = None) -> List[str]:
        """Produce the command line to encode the specified anchor's variant. Subclasses must supply its own implementation.
            Producing a reconstruction file while encoding is not enforced, 
                and when implemented must generate the appropriate json metadata (anchor.ReconstructionMeta) to support the metrics computation workflow.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def get_decoder_cmd(cls, bitstream: Path, reconstructed: Path, a: AnchorTuple) -> List[str]:
        """Produce the command line to decoder the specified anchor's variant. Subclasses must supply its own implementation.
        """
        raise NotImplementedError()

    @classmethod
    def decoder_log_metrics(cls, log: Path) -> dict:
        """parse decoder logs / stat files, and return a metrics dict
        """
        print(f'/!\\ decoder {cls} does not implement a custom log parsing')
        return { Metric.DECODETIME: 0 }

    @classmethod
    def encoder_log_metrics(cls, log: Path) -> dict:
        """parse encoder logs / stats, and return a metrics dict
        """
        print('/!\\ custom encoder does not implement log parsing')
        return {
            Metric.BITRATELOG: 0,
            Metric.ENCODETIME: 0
        }

    @classmethod
    def encode_variant(cls, a: AnchorTuple, variant_id: str, variant_qp: str, dst_dir: Path = None, dry_run = False) -> VariantData:
        """
        encode the variant and return VariantData
        """
        src = a.reference.path
        assert src.exists(), f'File not found: {src}'
        cfg = a.encoder_cfg
        assert cfg.exists(), f'File not found: {cfg}'

        if dst_dir is None:
            dst_dir = a.working_dir  # 'anchor working directory not found'
        dst_dir.mkdir(exist_ok=True, parents=True)

        a.working_dir.mkdir(parents=True, exist_ok=True)
        
        ENC = str(a.encoder_id).replace("-", "").replace(".", "_")
        encoder = os.getenv(f'{ENC}_ENCODER', cls.encoder_bin)
                
        bitstream = dst_dir / f'{variant_id}.bin'
        reconstruction = dst_dir / f'{variant_id}.yuv'
        logfile = dst_dir / f'{variant_id}.encoder.log'
        
        cmd = cls.get_encoder_cmd(a, variant_qp, bitstream, reconstruction)
        
        cfg_dir = Path(encoder).parent
        run_process(logfile, encoder, *cmd, dry_run=dry_run, cwd=cfg_dir)
        assert dry_run or reconstruction.exists(), 'reconstruction not found'

        dist = VideoSequence(reconstruction, **a.reference.properties)
        dist.start_frame = 1
        dist.frame_count = a.reference.frame_count
        coded_bit_depth = cls.get_encoding_bitdepth(a.encoder_cfg)
        if coded_bit_depth:
            dist.bit_depth = coded_bit_depth
        else:
            dist.bit_depth = a.reference.bit_depth

        dist.dump(dst_dir / f'{variant_id}.yuv.json')

        rec = ReconstructionMeta(
            a.encoder_id,
            reconstruction,
            None,
            md5=reconstruction.exists()
        )
        return VariantData.new(a, variant_id, variant_qp, logfile, bitstream, rec)

    @classmethod
    def decode_variant(cls, a: AnchorTuple, v: VariantData, dst_dir: Path = None, md5=True, dry_run=False) -> ReconstructionMeta:
        if dst_dir is not None:
            assert dry_run or dst_dir.is_dir(), f'decoder output directory not found: {dst_dir}'
        else:
            dst_dir = a.working_dir

        DEC = str(a.encoder_id).replace("-", "").replace(".", "_")
        decoder = os.getenv(f'{DEC}_DECODER', cls.decoder_bin)

        bitstream = a.working_dir / f'{v.variant_id}.bin'
        bitstream_no_sei, _ = cls.remove_sei(bitstream)
        if bitstream_no_sei is None:
            bitstream_no_sei = bitstream

        reconstructed = dst_dir / f'{bitstream.stem}.yuv'
        logfile = dst_dir / f'{bitstream.stem}.dec.log'
        
        cmd = cls.get_decoder_cmd(bitstream_no_sei, reconstructed, a)
        run_process(logfile, decoder, *cmd, dry_run=dry_run)
        
        dist = VideoSequence(reconstructed, **a.reference.properties)
        dist.start_frame = 1
        dist.frame_count = a.reference.frame_count
        coded_bit_depth = cls.get_encoding_bitdepth(a.encoder_cfg)
        dist.bit_depth = coded_bit_depth
        # if not dry_run:
        dist.dump(dst_dir / f'{v.variant_id}.yuv.json')
        return ReconstructionMeta(cls.encoder_id, reconstructed, logfile, md5=md5)

    @classmethod
    def get_encoding_bitdepth(cls, cfg:Path) -> int:
        # defaulty implem based on HM
        with open(cfg, 'r') as fo:
            for line in fo:
                m = re.match(r'(InternalBitDepth\s*:\s*)(\d*)', line)
                if m:
                    return int(m[2])
        raise Exception(f'{cls.encoder_id} - failed to parse coded bit depth in: {cfg}')


    @classmethod
    def remove_sei(cls, bitstream_in:Path, bitstream_out:Path=None):
        tool = cls.sei_removal_app
        if tool is None:
            return None, None
        if bitstream_out is None:
            bitstream_out = bitstream_in.with_suffix('.seirm.bin')
        cmd = [tool, '-b', str(bitstream_in), '-o', str(bitstream_out), '--DiscardPrefixSEI=1', '--DiscardSuffixSEI=1']
        log = bitstream_out.with_suffix('.log')
        run_process(log, *cmd, dry_run=False)
        return bitstream_out, log


    @classmethod
    def bitstream_size(cls, bitstream: Path) -> int:
        tmp, log = cls.remove_sei(bitstream)
        if tmp is None:
            s = int(os.path.getsize(bitstream))
        else:
            s = int(os.path.getsize(tmp))
            os.remove(tmp)
            os.remove(log)
        return s


def get_encoding_bitdepth(a: AnchorTuple) -> int:
    return get_encoder(a.encoder_id).get_encoding_bitdepth(a.encoder_cfg)


def _to_cli_args(opts: dict):
    """convert dict to cli options for HM & VTM softwares
    """
    args = []
    for k, v in opts.items():
        if k.startswith('--'):
            args.append(f'{k}={v}')
        else:
            args += [k, str(v)]
    return args



def reference_encoder_args(a: AnchorTuple, bitstream: Path, reconstruction: Path = None):
    """HM & VTM softwares share a common set of options
    """
    cmd = {
        "-c": f'{a.encoder_cfg}',
        "--InputFile": f'{a.reference.path}',
        "--BitstreamFile": str(bitstream),
        "--FrameRate": round(a.reference.frame_rate),
        "--FrameSkip": f'{a.start_frame -1}',
        "--FramesToBeEncoded": f'{a.frame_count}',
        "--SourceWidth": f'{a.reference.width}',
        "--SourceHeight": f'{a.reference.height}',
        "--InputBitDepth": f'{a.reference.bit_depth}',
        "--InputBitDepthC": f'{a.reference.bit_depth}',
        "--InputChromaFormat": f'{a.reference.chroma_subsampling.value}',
        "--MatrixCoefficients": a.reference.matrix_coefficients.value,
        "--TransferCharacteristics": a.reference.transfer_characteristics.value,
        "--ColourPrimaries": a.reference.colour_primaries.value,
        "--VideoFullRange": str(a.reference.video_full_range),
        "--ChromaSampleLocTypeTopField": f'{a.reference.chroma_sample_loc_type}',
        "--ChromaSampleLocTypeBottomField": f'{a.reference.chroma_sample_loc_type}',
        "--SEIDecodedPictureHash": "1",
    }

    if a.reference.hdr_master_display is not None:
        """
        # "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,47)"
        """
        p = r'([A-Z]+\(\d+\,\d+\))'
        q = r'([A-Z]+)\((\d+)\,(\d+)\)'
        parsed = re.findall(p, a.reference.hdr_master_display)
        parsed = [re.match(q, m).groups() for m in parsed]
        parsed = {k: (x,y) for (k, x, y) in parsed}
        cmd["SEIMasteringDisplayColourVolume"] = 1
        cmd["SEIMasteringDisplayPrimaries"] = ','.join([*parsed['G'], *parsed['B'], *parsed['R']])
        cmd["SEIMasteringDisplayMaxLuminance"] = parsed['L'][0] 
        cmd["SEIMasteringDisplayMinLuminance"] = parsed['L'][1]
        cmd["SEIMasteringDisplayWhitePoint"] = ','.join(parsed['WP'])
        
    if a.reference.hdr_max_cll is not None:
        cmd["--SEICLLEnabled"] = 1
        cmd["--SEICLLMaxContentLightLevel"] = a.reference.hdr_max_cll        
        
    if a.reference.hdr_max_fall is not None:
        cmd["--SEICLLEnabled"] = 1
        cmd["--SEICLLMaxPicAvgLightLevel"] = a.reference.hdr_max_fall

    if reconstruction is not None:
        cmd["--ReconFile"] = str(reconstruction)
    
    cmd["--PrintMSSSIM"] = 1
    cmd["--PrintHexPSNR"] = 1

    return _to_cli_args(cmd)


@register_encoder
class HM(EncoderBase):

    encoder_id = os.getenv("HM_VERSION", "HM")
    encoder_bin = "TAppEncoderStatic"
    decoder_bin = "TAppDecoderStatic"
    sei_removal_app = os.getenv("HM_SEI_REMOVAL_APP", "SEIRemovalAppStatic")


    @classmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        
        qp_args = ['-q', str(qp)]     
        
        if a.encoder_cfg_key in [
             # 6.2.8.3.2
             'S1-HM-01','S1-HM-02',
             # 6.3.8.3.2
             'S2-HM-01','S2-HM-02',
             # 6.5.8.3.2
             'S4-HM-02', 
             # 6.6.8.3.4, 6.6.8.3.6
             'S5-HM-02', 'S5-SCC-02'] :
            if a.reference.frame_rate <= 30:
                qp_args += ['--IntraPeriod=32'] 
            elif a.reference.frame_rate <= 60:
                qp_args += ['--IntraPeriod=64']
        
        elif a.encoder_cfg_key in ['S3-HM-02', 'S3-SCC-02']:
            # 6.4.8.3.4, 6.4.8.3.6
            qp_args += [f'--IntraPeriod={round(a.reference.frame_rate)}'] 
        return qp_args


    @classmethod
    def get_encoder_cmd(cls, a: AnchorTuple, variant_qp: str, bitstream: Path, reconstruction: Path = None) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction)
        args += cls.get_variant_cmd(a, variant_qp)
        rs = a.reference
        cls.get_encoding_bitdepth(a.encoder_cfg)
        level = None
        if (rs.width <= 2048) and (rs.frame_rate <= 30.) :
            level = "4"
        elif (rs.width <= 2048) and (rs.frame_rate <= 60.) :
            level = "4.1"
        elif (rs.width >= 3840) and (rs.frame_rate <= 30.) :
            level = "5"
        elif (rs.width >= 3840) and (rs.frame_rate <= 60.) :
            level = "5.1"
        else:
            logger.warn(f'can not determine H265 Level for {bitstream}')
        if level is not None:
            args += _to_cli_args({"--Level": level})
        return args
    
    @classmethod
    def get_decoder_cmd(cls, bitstream: Path, reconstructed: Path, a: AnchorTuple) -> List[str]:
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
        return [*_to_cli_args({"-b": f'{bitstream}', "-o": f'{reconstructed}', "-d": "0"})]

    @classmethod
    def encoder_log_metrics(cls, logp: Path) -> dict:
        summary = False
        keys = None
        values = None
        with open(logp, 'r') as logf:
            for line in logf.readlines():
                if summary:
                    if keys:
                        if not values:
                            values = line.split()[2:]  # '1', 'a'
                    else:
                        keys = line.split()[3:]  # 'Total', 'Frames', '|'
                else:
                    summary = line.startswith('SUMMARY')
                if values:
                    if line.startswith(' Total Time: '):
                        keys.append('EncodeTime')
                        values.append(line.split()[-2])

        assert summary and keys and values

        metrics = {}
        for i, k in enumerate(keys):
            if k == 'Bitrate':
                k = 'BitrateLog'
            elif k != 'EncodeTime':
                continue
            metrics[k] = float(values[i])

        return metrics


@register_encoder
class SCM(HM):

    encoder_id = os.getenv("SCM_VERSION", "SCM")
    encoder_bin = "TAppEncoderStatic"
    decoder_bin = "TAppDecoderStatic"
    sei_removal_app = os.getenv("HM_SEI_REMOVAL_APP", "SEIRemovalAppStatic")

    @classmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        # see base class function for GOP size 
        return super().get_variant_cmd(a, qp)

    @classmethod
    def get_encoder_cmd(cls, *args, **kwargs):
        return super().get_encoder_cmd(*args, **kwargs)

    @classmethod
    def get_decoder_cmd(cls, *args, **kwargs):
        return super().get_decoder_cmd(*args, **kwargs)


@register_encoder
class VTM(EncoderBase):

    encoder_id = os.getenv("VTM_VERSION", "VTM")
    encoder_bin = "EncoderAppStatic"
    decoder_bin = "DecoderAppStatic"
    sei_removal_app = os.getenv("VTM_SEI_REMOVAL_APP", "SEIRemovalAppStatic")

    @classmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        return ['-q', str(qp)]

    @classmethod
    def get_encoder_cmd(cls, a: AnchorTuple, variant_qp: str, bitstream: Path, reconstruction: Path = None) -> List[str]:
        args = reference_encoder_args(a, bitstream, reconstruction)
        args += cls.get_variant_cmd(a, variant_qp)
        # --LMCSSignalType         Input signal type: 0:SDR, 1:HDR-PQ, 2:HDR-HLG
        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream: Path, reconstructed: Path, a: AnchorTuple) -> List[str]:
        return _to_cli_args({"-b": f'{bitstream}', "-o": f'{reconstructed}', "--SEIDecodedPictureHash": "0" })


@register_encoder
class JM(EncoderBase):

    encoder_id = os.getenv("JM_VERSION", "JM")
    encoder_bin = "lencod_static"
    decoder_bin = "ldecod_static"

    @classmethod
    def remove_sei(cls, *args, **kwargs):
        return None, None
    
    @classmethod
    def get_encoding_bitdepth(cls, cfg:Path) -> int:
        # all configs have ProfileIDC = 100  (S3, S4, S5)
        return 8

    @classmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        quantization_cfg = {
            # 6.2.8.2.2	S1-JM-01
            'S1-JM-01': {
                '22': [19, 23, 23, "B7r0B3r3B1r4b0e6b2e6B5r4b4e6b6e6B11r3B9r4b8e6b10e6B13r4b12e6b14e6"],
                '27': [24, 28, 28, "B7r1B3r4B1r6b0e8b2e8B5r6b4e8b6e8B11r4B9r6b8e8b10e8B13r6b12e8b14e8"],
                '32': [29, 33, 33, "B7r2B3r5B1r7b0e8b2e8B5r7b4e8b6e8B11r5B9r7b8e8b10e8B13r7b12e8b14e8"],
                '37': [34, 38, 38, "B7r3B3r6B1r7b0e8b2e8B5r7b4e8b6e8B11r6B9r7b8e8b10e8B13r7b12e8b14e8"]
            },
            # 6.4.8.2.3	S3-JM-01: no Intra
            'S3-JM-01': {
                '22': [21, 24, None, "P0r3P1r2P2r3P3r2"],
                '27': [26, 29, None, "P0r3P1r2P2r3P3r2"],
                '32': [31, 34, None, "P0r6P1r5P2r6P3r5"],
                '37': [36, 39, None, "P0r6P1r5P2r6P3r5"],
                '42': [41, 44, None, "P0r6P1r5P2r6P3r5"]
            },
            # 6.4.8.2.3	S3-JM-02: fixed Intra every second
            'S3-JM-02': {
                '22': [22, 22, None, None],
                '27': [27, 27, None, None],
                '32': [32, 32, None, None],
                '37': [37, 37, None, None],
                '42': [42, 42, None, None]
            },
            # 6.5.8.2.3	S4-JM-01: no Intra, same parameters as for S5-JM-01 as documented in clause 6.6.8.2.3 are used.
            'S4-JM-01': {
                '22': [21, 24, 24, "B0r3B1r2B2r3B3r2"],
                '27': [26, 29, 29, "B0r5B1r4B2r5B3r4"],
                '32': [31, 34, 34, "B0r6B1r5B2r6B3r5"],
                '37': [36, 39, 39, "B0r6B1r5B2r6B3r5"]
            },
            # 6.5.8.2.4	S4-JM-02: Intra, same parameters as for S5-JM-02 as documented in clause 6.6.8.2.4 are used.
            'S4-JM-02': {
                '22': [22, 22, 22, None],
                '27': [27, 27, 27, None],
                '32': [32, 32, 32, None],
                '37': [37, 37, 37, None]
            },
            # 6.6.8.2.3	S5-JM-01: no Intra
            'S5-JM-01': {
                '22': [21, 24, 24, "B0r3B1r2B2r3B3r2"],
                '27': [26, 29, 29, "B0r5B1r4B2r5B3r4"],
                '32': [31, 34, 34, "B0r6B1r5B2r6B3r5"],
                '37': [36, 39, 39, "B0r6B1r5B2r6B3r5"]
            },
            # 6.6.8.2.4	S5-JM-02: Intra near 1 sec
            'S5-JM-02': {
                '22': [22, 22, 22, None],
                '27': [27, 27, 27, None],
                '32': [32, 32, 32, None],
                '37': [37, 37, 37, None]
            }
        }
        qpi, qpf, qpb, ehf = quantization_cfg[a.encoder_cfg_key][qp]
        qp_args = [
            '-p', f'QPISlice={qpi}',
            '-p', f'QPPSlice={qpf}'
        ]
        if qpb is not None:
            qp_args += ['-p', f'QPBSlice={qpb}']
        if ehf is not None:
            qp_args += ['-p', f'ExplicitHierarchyFormat={ehf}']
        

        # 6.2.8.2.2/6.5.8.2.4/6.6.8.2.4	IntraPeriod: power of 2 value that is greater than or equal to the frame rate such that near 1 second is achieved
        if a.encoder_cfg_key == 'S1-JM-01':
            if a.reference.frame_rate <= 30:
                qp_args += ['-p', 'IntraPeriod=32', '-p', 'IDRPeriod=0'] 
            elif a.reference.frame_rate <= 60:
                qp_args += ['-p', 'IntraPeriod=64', '-p', 'IDRPeriod=0']
        elif a.encoder_cfg_key in  ['S4-JM-02', 'S5-JM-02']:
            if a.reference.frame_rate <= 30:
                qp_args += ['-p', 'IntraPeriod=32', '-p', 'IDRPeriod=32'] 
            elif a.reference.frame_rate <= 60:
                qp_args += ['-p', 'IntraPeriod=64', '-p', 'IDRPeriod=64']
        elif a.encoder_cfg_key == 'S3-JM-02':
            # 6.4.8.2.3	S3-JM-02: fixed Intra every second
            gs = round(a.reference.frame_rate)
            qp_args += ['-p', f'IntraPeriod={gs}', '-p', f'IDRPeriod={gs}']
        return qp_args


    @classmethod
    def get_encoder_cmd(cls, a: AnchorTuple, variant_qp: str, bitstream: Path, reconstruction: Path = None) -> List[str]:
        
        ref = a.reference
        if str(a.encoder_cfg_key).lower() == 's1-jm-01':
            ref = as_8bit_sequence(a.reference)
        
        args = [
            "-d", str(a.encoder_cfg),
            "-p", f'InputFile={ref.path}',
            "-p", f'OutputFile={bitstream}',
            "-p", f'FrameRate={float(ref.frame_rate)}',
            "-p", f'StartFrame={a.start_frame -1}',
            "-p", f'FramesToBeEncoded={a.frame_count}',
            "-p", f'SourceWidth={ref.width}',
            "-p", f'SourceHeight={ref.height}',
            "-p", f'OutputWidth={ref.width}',
            "-p", f'OutputHeight={ref.height}',
            "-p", f'SourceBitDepthLuma={ref.bit_depth}',
            "-p", f'SourceBitDepthChroma={ref.bit_depth}'
        ]

        if ref.interleaved:
            args += ["-p", "Interleaved=1"]

        if int(ref.chroma_format == ChromaFormat.RGB):
            args += ["-p", 'RGBInput=1']
        else:
            args += ["-p", 'RGBInput=0']

        if ref.video_full_range == 1:
            args += ["-p", f'StandardRange=0']
        else:
            args += ["-p", f'StandardRange=1']

        if ref.chroma_subsampling == ChromaSubsampling.CS_400:
            args += ["-p", 'YUVFormat=0']
        elif ref.chroma_subsampling == ChromaSubsampling.CS_420:
            args += ["-p", 'YUVFormat=1']
        elif ref.chroma_subsampling == ChromaSubsampling.CS_422:
            args += ["-p", 'YUVFormat=2']
        elif ref.chroma_subsampling == ChromaSubsampling.CS_444:
            args += ["-p", 'YUVFormat=3']

        if reconstruction is not None:
            args += ["-p", f'ReconFile={reconstruction}']

        args += cls.get_variant_cmd(a, variant_qp)

        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream: Path, reconstructed: Path, a: AnchorTuple) -> List[str]:
        args = ["-i", f'{bitstream}', "-o", f'{reconstructed}']
        return args



@register_encoder
class SCC(SCM):
    encoder_id = os.getenv("SCC_VERSION", "SCC")


@register_encoder
class ETM(EncoderBase):

    encoder_id = os.getenv("ETM_VERSION", "ETM")
    encoder_bin = "evca_encoder"
    decoder_bin = "evca_decoder"
    # sei_removal_app = os.getenv("ETM_SEI_REMOVAL_APP", "SEIRemovalAppStatic")


    @classmethod
    def get_variant_cmd(cls, a: AnchorTuple, qp) -> List[str]:
        return ['-q', str(qp)]

    @classmethod
    def get_encoder_cmd(cls, a: AnchorTuple, variant_qp: str, bitstream: Path, reconstruction: Path = None) -> List[str]:
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

        assert a.reference.chroma_sample_loc_type == 2, 'chroma_sample_loc_type != 2 not implemented'

        if reconstruction:
            args += ['-r', str(reconstruction)]

        args += cls.get_variant_cmd(a, variant_qp)

        return args

    @classmethod
    def get_decoder_cmd(cls, bitstream: Path, reconstructed: Path, a: AnchorTuple) -> List[str]:
        opl = bitstream.with_suffix('.opl')
        bd = cls.get_encoding_bitdepth(a.encoder_cfg)
        args = [
            "-i", f'{bitstream}',
            "-o", f'{reconstructed}',
            "--opl", str(opl),
            "--output_bit_depth", str(bd),
            "-v", '1'  # 0=quiet, 2=verbose
        ]
        return args

    @classmethod
    def get_encoding_bitdepth(cls, cfg: Path) -> int:
        with open(cfg) as f:
            for l in f:
                if l.startswith('codec_bit_depth'):
                    return int(l.split('=')[-1])
        assert False, 'failed to parse codec_bit_depth in ETM encoder config'
