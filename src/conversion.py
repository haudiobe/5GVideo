from enum import Enum
import os
import copy
from pathlib import Path
from typing import List
from utils import run_process, VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from concurrent.futures import ProcessPoolExecutor
from encoders import parse_encoding_bitdepth
from anchor import AnchorTuple, AnchorTupleCtx


class Conversion(Enum):
    NONE = 0
    HDRCONVERT_8TO10BIT = 1
    HDRCONVERT_YCBR420TOEXR2020 = 2

def get_anchor_conversion_type(a:AnchorTuple) -> Conversion:
    if a.reference.hdr_master_display:
        return Conversion.HDRCONVERT_YCBR420TOEXR2020
    coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
    if (a.reference.bit_depth == 8) and (coded_bit_depth == 10):
        return Conversion.HDRCONVERT_8TO10BIT
    return Conversion.NONE
 

# see: please refer to HDRConvert.cfg for the full range of options
def __color_primaries(v: VideoSequence):
    if v.colour_primaries == ColorPrimaries.BT_709:
        return 0
    if v.colour_primaries == ColorPrimaries.BT_2020:
        return 1
    assert False, f'ColorPrimaries: {v.colour_primaries} - has not been tested'


def __color_space(v: VideoSequence):
    if v.chroma_format == ChromaFormat.YUV:
        return 0
    elif v.chroma_format == ChromaFormat.RGB:
        return 1

    raise Exception('Unknown video sequence chroma format')


def __chroma_format(v: VideoSequence):
    if v.chroma_subsampling == ChromaSubsampling.CS_420:
        return 1
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        return 2
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        return 3
    elif v.chroma_subsampling == ChromaSubsampling.CS_400:
        return 0
    raise Exception('Unknown video sequence chroma subsampling')


def __transfer_function(v: VideoSequence):
    if v.transfer_characteristics in [
            TransferFunction.BT709,
            TransferFunction.BT2020_SDR]:
        return 0  # no TF
    elif v.chroma_subsampling == TransferFunction.BT2020_HLG:
        return 3
    elif v.chroma_subsampling == TransferFunction.BT2020_PQ:
        return 1
    return 0


def __sample_range(v: VideoSequence):
    sr = int(v.video_full_range)
    assert sr in [0, 1], 'unexpected sample range'
    return sr


def __pixel_format(v: VideoSequence) -> int:
    """FourCC code
        0: UYVY
        1: YUY2
        2: YUYV
        3: YVYU
        4: BGR
        5: RGB
        6: V210
        7: UYVY10
        8: V410
        9: R210
        10: R10K
        11: XYZ
    """
    assert not v.interleaved, "unspecified pixel format for interleaved data"
    return 6 # 2 


def conversion_path(sequence: Path, suffix: str) -> Path:
    return sequence.parent / f'{sequence.stem}.conv-{suffix}'


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
        '-p', f'SourceChromaFormat={__chroma_format(yuv_in)}',
        '-p', f'SourceFourCCCode={__pixel_format(yuv_in)}',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={__color_space(yuv_in)}',
        '-p', f'SourceColorPrimaries={__color_primaries(yuv_in)}',
        '-p', f'SourceSampleRange={__sample_range(yuv_in)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceChromaLocationTop={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceChromaLocationBottom={int(yuv_in.chroma_sample_loc_type)}',
        # Output 
        '-p', f'OutputChromaFormat={__chroma_format(yuv_out)}',
        '-p', f'OutputColorSpace={__color_space(yuv_out)}',
        '-p', f'OutputColorPrimaries={__color_primaries(yuv_out)}',
        '-p', f'OutputSampleRange={__color_primaries(yuv_out)}',
        '-p', f'OutputTransferFunction={__transfer_function(yuv_out)}',
        '-p', f'OutputSampleRange={__sample_range(yuv_out)}',  # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',
        '-p', f'OutputInterleaved={int(yuv_out.interleaved)}',
        '-p', f'OutputFourCCCode={__pixel_format(yuv_in)}',
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
    return exr_out


def hdr_convert_cmd_YCbCr420toExr2020(yuv_in: VideoSequence, exr_out: VideoSequence, logfile: Path = None) -> List[str]:

    opts = [
        '-p', f'SourceFile={yuv_in.path}',
        '-p', f'OutputFile={exr_out.path}',

        # Input 
        '-p', f'SourceWidth={yuv_in.width}',
        '-p', f'SourceHeight={yuv_in.height}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceChromaFormat={__chroma_format(yuv_in)}',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        # 4CC needs review ############################################################
        '-p', f'SourceFourCCCode={__pixel_format(yuv_in)}',
        # '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        ###############################################################################
        '-p', f'SourceColorSpace={__color_space(yuv_in)}', # 0 CM_YUV
        '-p', f'SourceColorPrimaries={__color_primaries(yuv_in)}', # 1 BT.2020
        '-p', f'SourceSampleRange={__sample_range(yuv_in)}', # 0 Standard
        '-p', f'SourceTransferFunction{__transfer_function(yuv_in)}', # 1 PQ
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
        '-p', f'OutputSampleRange=1',  # Full range
        '-p', 'OutputTransferFunction=0',

        '-p', f'NumberOfFrames={yuv_in.frame_count}',
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', 'InputFileHeader=0',  # header bytes count
        '-p', 'FrameSkip=0'  # seems redundant with StartFrame, unclear how HDRtools uses it.
    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts


def convert_sequence(conv:Conversion, input: VideoSequence, dry_run=False, log=False):
    print("convert_sequence !!!!!")

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

