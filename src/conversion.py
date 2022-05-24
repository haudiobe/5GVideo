from enum import Enum
import os
from pathlib import Path
from typing import List
from anchor import AnchorTuple
from encoders import get_encoding_bitdepth
from sequences import VideoSequence, as_8bit_sequence, as_10bit_sequence, as_exr2020_sequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from utils import run_process
from typing import Tuple

class Conversion(Enum):
    NONE = 0
    HDRCONVERT_8TO10BIT = 1
    HDRCONVERT_YCBR420TOEXR2020 = 2
    HDRCONVERT_10TO8BIT = 3


def get_anchor_conversion_type(a:AnchorTuple) -> Tuple[Conversion, Conversion]:
    if a.reference.transfer_characteristics is None:
        raise Exception('can not define conversion for sequence with invalid metadata')
        
    if a.reference.transfer_characteristics == TransferFunction.BT2020_PQ:
        return (Conversion.HDRCONVERT_YCBR420TOEXR2020, Conversion.HDRCONVERT_YCBR420TOEXR2020)

    if str(a.encoder_cfg_key).lower() == 's1-jm-01':
        return (Conversion.HDRCONVERT_10TO8BIT, Conversion.HDRCONVERT_8TO10BIT)

    ref = Conversion.NONE

    assert a.reference.bit_depth is not None
    if (a.reference.bit_depth == 8):
        ref = Conversion.HDRCONVERT_8TO10BIT
    else:
        assert a.reference.bit_depth == 10

    recd = get_encoding_bitdepth(a) # assumes reconstruction has same bitdepth as coded bitdepth
    if recd == 10:
        return (ref, Conversion.NONE)
    else:
        assert recd == 8
        return (ref, Conversion.HDRCONVERT_8TO10BIT)
    

def hdrtools_color_primaries(v: VideoSequence):
    if v.colour_primaries == ColorPrimaries.BT_709:
        return 0
    if v.colour_primaries == ColorPrimaries.BT_2020:
        return 1
    assert False, f'ColorPrimaries: {v.colour_primaries} - has not been tested'


def hdrtools_color_space(v: VideoSequence):
    if v.chroma_format == ChromaFormat.YUV:
        return 0
    elif v.chroma_format == ChromaFormat.RGB:
        return 1

    raise Exception('Unknown video sequence chroma format')


def hdrtools_chroma_format(v: VideoSequence):
    if v.chroma_subsampling == ChromaSubsampling.CS_420:
        return 1
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        return 2
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        return 3
    elif v.chroma_subsampling == ChromaSubsampling.CS_400:
        return 0
    raise Exception('Unknown video sequence chroma subsampling')


def hdrtools_transfer_function(v: VideoSequence):
    if v.transfer_characteristics in [
            TransferFunction.BT709,
            TransferFunction.BT2020_SDR]:
        return 0  # no TF
    elif v.transfer_characteristics == TransferFunction.BT2020_HLG:
        return 3
    elif v.transfer_characteristics == TransferFunction.BT2020_PQ:
        return 1
    raise ValueError('Invalid tranfer function')


def hdrtools_sample_range(v: VideoSequence):
    sr = int(v.video_full_range)
    assert sr in [0, 1], 'unexpected sample range'
    return sr


def hdrtools_pixel_format(v: VideoSequence) -> int:
    assert not v.interleaved, "unspecified pixel format for interleaved data"
    return 6 


def hdr_convert_cmd_10to8bits(yuv_in: VideoSequence, yuv_out: VideoSequence, file_header=0, quiet=False, logfile: Path = None) -> List[str]:
    assert yuv_in.width == yuv_out.width, "resizing not supported, in/out width must match"
    assert yuv_in.height == yuv_out.height, "resizing not supported, in/out width must match"
    assert yuv_in.frame_rate == yuv_out.frame_rate

    cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) /  'HDRConvert_10bto8b.cfg'
    assert cfg.exists(), f'config file not found: {cfg}'
    
    opts = [
        '-f', str(cfg),

        '-p', f'SourceFile={yuv_in.path}',
        '-p', f'OutputFile={yuv_out.path}',
        '-p', f'SilentMode={int(quiet)}',

        # Input 
        '-p', f'SourceWidth={yuv_in.width}',
        '-p', f'SourceHeight={yuv_in.height}',
        '-p', f'SourceChromaFormat={hdrtools_chroma_format(yuv_in)}',
        # '-p', f'SourceFourCCCode={hdrtools_pixel_format(yuv_in)}',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={hdrtools_color_space(yuv_in)}',
        '-p', f'SourceColorPrimaries={hdrtools_color_primaries(yuv_in)}',
        '-p', f'SourceSampleRange={hdrtools_sample_range(yuv_in)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceChromaLocationTop={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceChromaLocationBottom={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceTransferFunction=12', # NORMALIZE
        # Output 
        '-p', f'OutputTransferFunction=12',
        '-p', f'OutputChromaFormat={hdrtools_chroma_format(yuv_out)}',
        '-p', f'OutputColorSpace={hdrtools_color_space(yuv_out)}',
        '-p', f'OutputColorPrimaries={hdrtools_color_primaries(yuv_out)}',
        '-p', f'OutputSampleRange={hdrtools_sample_range(yuv_out)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',
        '-p', f'OutputInterleaved={int(yuv_out.interleaved)}',
        '-p', f'NumberOfFrames={yuv_in.frame_count}',  # frames to process, -1 auto based on input file
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', 'InputFileHeader=0',  # header bytes count
        '-p', 'FrameSkip=0',  # seems redundant with StartFrame, unclear how HDRtools uses it.

    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts


def hdr_convert_cmd_8to10bits(yuv_in: VideoSequence, yuv_out: VideoSequence, file_header=0, quiet=False, logfile: Path = None ) -> List[str]:
    assert yuv_in.width == yuv_out.width, "resizing not supported, in/out width must match"
    assert yuv_in.height == yuv_out.height, "resizing not supported, in/out width must match"
    assert yuv_in.frame_rate == yuv_out.frame_rate

    cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) / 'HDRConvert_8bto10b.cfg'
    assert cfg.exists(), f'config file not found: {cfg}'
    
    opts = [
        '-f', str(cfg),

        '-p', f'SourceFile={yuv_in.path}',
        '-p', f'OutputFile={yuv_out.path}',
        '-p', f'SilentMode={int(quiet)}',

        # Input 
        '-p', f'SourceWidth={yuv_in.width}',
        '-p', f'SourceHeight={yuv_in.height}',
        '-p', f'SourceChromaFormat={hdrtools_chroma_format(yuv_in)}',
        # '-p', f'SourceFourCCCode={hdrtools_pixel_format(yuv_in)}',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={hdrtools_color_space(yuv_in)}',
        '-p', f'SourceColorPrimaries={hdrtools_color_primaries(yuv_in)}',
        '-p', f'SourceSampleRange={hdrtools_sample_range(yuv_in)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceChromaLocationTop={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceChromaLocationBottom={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceTransferFunction=12', # NORMALIZE
        # Output 
        '-p', f'OutputTransferFunction=12',
        '-p', f'OutputChromaFormat={hdrtools_chroma_format(yuv_out)}',
        '-p', f'OutputColorSpace={hdrtools_color_space(yuv_out)}',
        '-p', f'OutputColorPrimaries={hdrtools_color_primaries(yuv_out)}',
        '-p', f'OutputSampleRange={hdrtools_sample_range(yuv_out)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',
        '-p', f'OutputInterleaved={int(yuv_out.interleaved)}',
        '-p', f'NumberOfFrames={yuv_in.frame_count}',  # frames to process, -1 auto based on input file
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', 'InputFileHeader=0',  # header bytes count
        '-p', 'FrameSkip=0',  # seems redundant with StartFrame, unclear how HDRtools uses it.

    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts


def hdr_convert_cmd_YCbCr420toExr2020(yuv_in: VideoSequence, exr_out: VideoSequence, logfile: Path = None ) -> List[str]:
    
    cfg = Path(os.getenv('HDRMETRICS_CFG_DIR', '/home/cfg')) / 'HDRConvertYCbCr420ToEXR2020.cfg'
    assert cfg.exists(), f'config file not found: {cfg}'
    
    opts = [
        '-f', str(cfg),

        '-p', f'SourceFile={yuv_in.path}',
        '-p', f'OutputFile={exr_out.path}',

        # Input 
        '-p', f'SourceWidth={yuv_in.width}',
        '-p', f'SourceHeight={yuv_in.height}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceChromaFormat={hdrtools_chroma_format(yuv_in)}',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={hdrtools_color_space(yuv_in)}', # 0 CM_YUV
        '-p', f'SourceColorPrimaries={hdrtools_color_primaries(yuv_in)}', # 1 BT.2020
        '-p', f'SourceSampleRange={hdrtools_sample_range(yuv_in)}', # 0 Standard
        '-p', f'SourceTransferFunction={hdrtools_transfer_function(yuv_in)}', # 1 PQ
        '-p', f'SourceChromaLocationTop={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceChromaLocationBottom={int(yuv_in.chroma_sample_loc_type)}',
        
        # Output 
        '-p', f'OutputRate={yuv_in.frame_rate}',
        '-p', 'OutputChromaFormat=3',  # 444
        '-p', 'OutputBitDepthCmp0=10',
        '-p', 'OutputBitDepthCmp1=10',
        '-p', 'OutputBitDepthCmp2=10',
        '-p', 'OutputColorSpace=1',  # CM_RGB
        '-p', 'OutputColorPrimaries=1',  # BT.2020
        '-p', 'OutputSampleRange=1',  # Full range
        '-p', 'OutputTransferFunction=0', # no TF
        '-p', f'NumberOfFrames={yuv_in.frame_count}',
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', 'InputFileHeader=0',  # header bytes count
        '-p', 'FrameSkip=0'  # seems redundant with StartFrame, unclear how HDRtools uses it.
    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts


def convert_sequence(conv:Conversion, input: VideoSequence, dry_run=False, log=False):

    HDRCONVERT_TOOL = os.getenv("HDRCONVERT_TOOL", "HDRConvert")
    assert HDRCONVERT_TOOL, "/!\\ missing 'HDRCONVERT_TOOL' environment variable - path to HDRConvert executable"

    output, conv_fn = None, None

    if conv == Conversion.HDRCONVERT_10TO8BIT:
        output = as_8bit_sequence(input)
        conv_fn = hdr_convert_cmd_10to8bits

    elif conv == Conversion.HDRCONVERT_8TO10BIT:
        output = as_10bit_sequence(input)
        conv_fn = hdr_convert_cmd_8to10bits

    elif conv == Conversion.HDRCONVERT_YCBR420TOEXR2020:
        output = as_exr2020_sequence(input)
        conv_fn = hdr_convert_cmd_YCbCr420toExr2020

    else:
        raise ValueError(f'Invalid conversion type {conv}')
    
    fp = output.path.with_suffix('.json')
    output.dump(fp)
    logfile = output.path.with_suffix('.hdrconvert.log')
    cmd = conv_fn(input, output)
    run_process(logfile, HDRCONVERT_TOOL, *cmd, dry_run=dry_run)

