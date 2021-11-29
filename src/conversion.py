from enum import Enum
import os
import copy
from pathlib import Path
from typing import List
from utils import run_process, VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from encoders import get_encoding_bitdepth
from anchor import AnchorTuple

class Conversion(Enum):
    NONE = 0
    HDRCONVERT_8TO10BIT = 1
    HDRCONVERT_YCBR420TOEXR2020 = 2

def get_anchor_conversion_type(a:AnchorTuple) -> Conversion:
    if a.reference.transfer_characteristics == TransferFunction.BT2020_PQ:
        return Conversion.HDRCONVERT_YCBR420TOEXR2020
    # compute all metrics on 10 bit assets, even for JM
    if (a.reference.bit_depth == 8):
        return Conversion.HDRCONVERT_8TO10BIT
    assert get_encoding_bitdepth(a) == a.reference.bit_depth
    return Conversion.NONE
 

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


def conversion_path(sequence: Path, suffix: str) -> Path:
    return sequence.parent / 'tmp' / f'{sequence.stem}{suffix}'


# return a modified VideoSequence obj pointing to the 10bit conversion
def as_10bit_sequence(yuv_in: VideoSequence) -> VideoSequence:
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 10
    yuv_out.path = conversion_path(yuv_out.path, '10bit.yuv')
    return yuv_out


def hdr_convert_cmd_8to10bits(yuv_in: VideoSequence, yuv_out: VideoSequence, file_header=0, quiet=False, logfile: Path = None) -> List[str]:
    assert yuv_in.width == yuv_out.width, "resizing not supported, in/out width must match"
    assert yuv_in.height == yuv_out.height, "resizing not supported, in/out width must match"
    assert yuv_in.frame_rate == yuv_out.frame_rate
    
    opts = [
        '-p', f'SourceFile={yuv_in.path}',
        '-p', f'OutputFile={yuv_out.path}',
        '-p', f'SilentMode={int(quiet)}',

        # Input 
        '-p', f'SourceWidth={yuv_in.width}',
        '-p', f'SourceHeight={yuv_in.height}',
        '-p', f'SourceChromaFormat={hdrtools_chroma_format(yuv_in)}',
        '-p', f'SourceFourCCCode={hdrtools_pixel_format(yuv_in)}',
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
        # Output 
        '-p', f'OutputChromaFormat={hdrtools_chroma_format(yuv_out)}',
        '-p', f'OutputColorSpace={hdrtools_color_space(yuv_out)}',
        '-p', f'OutputColorPrimaries={hdrtools_color_primaries(yuv_out)}',
        '-p', f'OutputTransferFunction={hdrtools_transfer_function(yuv_out)}',
        '-p', f'OutputSampleRange={hdrtools_sample_range(yuv_out)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',
        '-p', f'OutputInterleaved={int(yuv_out.interleaved)}',
        '-p', f'OutputFourCCCode={hdrtools_pixel_format(yuv_in)}',
        '-p', f'NumberOfFrames={yuv_in.frame_count}',  # frames to process, -1 auto based on input file
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', 'InputFileHeader=0',  # header bytes count
        '-p', 'FrameSkip=0',  # seems redundant with StartFrame, unclear how HDRtools uses it.

        # explicit defaults ...
        # 444 to 420 conversion filters
        '-p', 'ChromaDownsampleFilter=2',
        # 420 to 444 conversion filters
        '-p', 'ChromaUpsampleFilter=1',
        # OpenEXR output file precision, 0: HALF, 1: SINGLE
        '-p', 'SetOutputSinglePrec=0',
        # Enable rounding for EXR outputs
        '-p', 'SetOutputEXRRounding=0'
    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts


def as_exr2020_sequence(yuv_in: VideoSequence) -> VideoSequence:
    exr_out = copy.deepcopy(yuv_in)
    exr_out.bit_depth = 10
    exr_out.chroma_subsampling = ChromaSubsampling.CS_444
    exr_out.path = conversion_path(exr_out.path, '_2020_444_%05d.exr')
    exr_out.chroma_subsampling = ChromaSubsampling.CS_444
    exr_out.chroma_format = ChromaFormat.RGB
    exr_out.transfer_characteristics = TransferFunction.NONE
    exr_out.colour_primaries = ColorPrimaries.BT_2020
    exr_out.video_full_range = 1
    return exr_out


def hdr_convert_cmd_YCbCr420toExr2020(yuv_in: VideoSequence, exr_out: VideoSequence, logfile: Path = None, cfg = '/home/cfg/HDRConvertYCbCr420ToEXR2020.cfg',) -> List[str]:
    
    opts = [
        '-f', cfg,

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

    HDRCONVERT_TOOL = os.getenv("HDRCONVERT_TOOL", None)
    assert HDRCONVERT_TOOL, "/!\\ missing 'HDRCONVERT_TOOL' environment variable - path to HDRConvert executable"

    output, conv_fn = None, None

    if conv == Conversion.HDRCONVERT_8TO10BIT:
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

