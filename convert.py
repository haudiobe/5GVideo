#!.venv/bin/python3

import argparse, os, copy
from pathlib import Path
from utils import run_process, VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from encoders import parse_encoding_bitdepth
from anchor import AnchorTuple, VariantData
from download import AnchorTupleCtx

# see: please refer to HDRConvert.cfg for the full range of options

def __color_primaries(v:VideoSequence):
    
    if v.colour_primaries == ColorPrimaries.BT_709:
        return 0
    if v.colour_primaries == ColorPrimaries.BT_2020:
        return 1
    assert False, f'ColorPrimaries: {v.colour_primaries} - has not been tested'

def __color_space(v:VideoSequence):
    if v.chroma_format == ChromaFormat.YUV:
        return 0
    elif v.chroma_format == ChromaFormat.RGB:
        return 1

    raise Exception('Unknown video sequence chroma format')
    
def __chroma_format(v:VideoSequence):
    if v.chroma_subsampling == ChromaSubsampling.CS_420:
        return 1
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        return 2
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        return 3
    elif v.chroma_subsampling == ChromaSubsampling.CS_400:
        return 0
    raise Exception('Unknown video sequence chroma subsampling')

def __transfer_function(v:VideoSequence):
    if v.transfer_characteristics in [
            TransferFunction.BT709,
            TransferFunction.BT2020_SDR
        ]:
        return 0 # no TF
    elif v.chroma_subsampling == TransferFunction.BT2020_HLG:
        return 3
    elif v.chroma_subsampling == TransferFunction.BT2020_PQ:
        return 1
    return 0

def __sample_range(v:VideoSequence):
    sr = int(v.video_full_range)
    assert sr in [0, 1], f'unexpected sample range'
    return sr


def __pixel_format(v:VideoSequence):
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
    return 2 

def conversion_path(sequence:Path, suffix:str) -> Path:
    return sequence.parent / f'{sequence.stem}.conv-{suffix}.yuv'

# return a modified VideoSequence obj pointing to the 10bit conversion
def as_10bit_sequence(yuv_in:VideoSequence) -> VideoSequence :
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 10
    yuv_out.path = conversion_path(yuv_out.path, '10bit')
    return yuv_out

# return a modified VideoSequence obj pointing to the 10bit conversion
def as_8bit_sequence(yuv_in:VideoSequence) -> VideoSequence :
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 8
    yuv_out.path = conversion_path(yuv_out.path, '8bit')
    return yuv_out

def hdr_convert_cmd(yuv_in:VideoSequence, yuv_out:VideoSequence, file_header=0, quiet=False, logfile:Path=None):
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
        # '-p', 'SourceFourCCCode=2',
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={__color_space(yuv_in)}',
        '-p', f'SourceColorPrimaries={__color_primaries(yuv_in)}',
        '-p', f'SourceSampleRange={__sample_range(yuv_in)}', # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        '-p', f'SourceRate={yuv_in.frame_rate}',
        '-p', f'SourceChromaLocationTop={int(yuv_in.chroma_sample_loc_type)}',
        '-p', f'SourceChromaLocationBottom={int(yuv_in.chroma_sample_loc_type)}',
        # SourceTransferFunction=0  # Transfer Function
        #                           0: NULL (no new TF applied)
        #                           1: PQ
        #                           2: PH
        #                           3: Hybrid Gamma TF
        #                           4: Adaptive PQ
        #                           5: Adaptive PH
        #                           6: Power Law Gamma
        # SourceDisplayAdjustment=0    # Apply a Gamma adjustment to the source
        # SourceSystemGamma=1.0
        # SourceTransferMinBrightness=0.0     # Transfer Function Minimum Brightness
        # SourceTransferMaxBrightness=10000.0 # Transfer Function Maximum Brightness
        # OutputTransferMinBrightness=0.0     # Transfer Function Minimum Brightness
        # OutputTransferMaxBrightness=10000.0 # Transfer Function Maximum Brightness
        # SourceConstantLuminance=0
        # OutputConstantLuminance=0

        # Output. Note: changing dimensions isn't implemented 
        '-p', f'OutputChromaFormat={__chroma_format(yuv_out)}',
        '-p', f'OutputColorSpace={__color_space(yuv_out)}',
        '-p', f'OutputColorPrimaries={__color_primaries(yuv_out)}',
        '-p', f'OutputSampleRange={__color_primaries(yuv_out)}',
        '-p', f'OutputTransferFunction={__transfer_function(yuv_out)}',
        '-p', f'OutputSampleRange={__sample_range(yuv_out)}', # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',
        '-p', f'OutputInterleaved={int(yuv_out.interleaved)}',
        '-p', f'OutputFourCCCode={__pixel_format(yuv_in)}',
        # '-p', f'OutputFourCCCode=2',
        '-p', f'NumberOfFrames={yuv_in.frame_count}', # frames to process, -1 auto based on input file
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', f'InputFileHeader=0', # header bytes count
        '-p', f'FrameSkip=0', # seems redundant with StartFrame, unclear how HDRtools uses it.

        # explicit defaults ...
        # 444 to 420 conversion filters
        '-p', f'ChromaDownsampleFilter=2',
        # 420 to 444 conversion filters
        '-p', f'ChromaUpsampleFilter=1',
        # OpenEXR output file precision, 0: HALF, 1: SINGLE
        '-p', f'SetOutputSinglePrec=0',
        # Enable rounding for EXR outputs
        '-p', f'SetOutputEXRRounding=0'
    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts

def convert(yuv_in:VideoSequence, yuv_out:VideoSequence=None, quiet=False, dry_run=False, logfile=None):

    HDRCONVERT_TOOL = os.getenv("HDRCONVERT_TOOL", None)
    assert HDRCONVERT_TOOL, "/!\\ missing 'HDRCONVERT_TOOL' environment variable - path to HDRConvert executable"

    cmd = hdr_convert_cmd(yuv_in, yuv_out, quiet=quiet, logfile=None)
    logfile = yuv_out.path.with_suffix('.hdrconvert.log')
    run_process(logfile, HDRCONVERT_TOOL, *cmd, dry_run=dry_run)

def can_compute_metrics(a:AnchorTuple, vd:VariantData) -> bool:
    coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
    if (a.reference.bit_depth == 8) and (coded_bit_depth == 10):
        vc = as_10bit_sequence(a.reference)
        return vc.path.exists()


def main():
    ctx = AnchorTupleCtx.parse_args()
    assert ctx.scenario_dir.exists(), f'Not found: {ctx.scenario_dir}'
    assert len(ctx.scenario_dir.parts) >= 2, f'invalid scenario directory: {ctx.scenario_dir}'
    scenario = ctx.scenario_dir.parts[-2]
    codec = ctx.scenario_dir.parts[-1]
    base_dir = ctx.scenario_dir.parent.parent.parent
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        
        _8to10bit = set()
        futures = []

        for a in ctx.iter_anchors():

            # convert 8bit ref sequences to 10bit for HDR tools (scenario-3 & scenario-5)
            coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
            if (a.reference.bit_depth == 8) and (coded_bit_depth == 10):
                if str(a.reference.path) in _8to10bit:
                    continue
                yuv_in = a.reference
                yuv_out = as_10bit_sequence(a.reference)
                fp = yuv_out.path.with_suffix('.json')
                if fp.exists() and yuv_out.path.exists():
                    r = yuv_out.path.stat().st_size / yuv_in.path.stat().st_size
                    if r != 2.0:
                        print(f'[overwritting] {yuv_out.path}')
                    else:
                        print(f'[conversion already exists] {yuv_out.path}')
                        _8to10bit.add(str(a.reference.path))
                        continue
                else:
                    print(f'[missing converion] {yuv_out.path}')

                if not ctx.dry_run:
                    yuv_out.dump(fp)
                
                fut = executor.submit(convert, yuv_in, yuv_out, quiet=True, dry_run=ctx.dry_run)
                futures.append(fut)
                _8to10bit.add(str(a.reference.path))

        executor.shutdown(wait=True)
        for fut in futures:
            try:
                r = fut.result()
                if r:
                    print(r)
            except BaseException as e:
                print('ERROR:', e)



if __name__ == "__main__":
    main()