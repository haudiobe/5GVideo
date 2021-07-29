#!.venv/bin/python3

import argparse, os, copy
from pathlib import Path
from utils import run_process, VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from encoders import parse_encoding_bitdepth

from download import AnchorTupleCtx

def __color_primaries(v:VideoSequence):
    
    if v.colour_primaries == ColorPrimaries.BT_709:
        return 0
    if v.colour_primaries == ColorPrimaries.BT_2020:
        return 1
    assert False, f'ColorPrimaries: {v.colour_primaries} - has not been tested'

def __color_space(v:VideoSequence):
    """
    typedef enum {
        CM_UNKNOWN   = -1,
        CM_YCbCr     =  0,
        CM_RGB       =  1,
        CM_XYZ       =  2,
        CM_YDZDX     =  3,
        CM_YCbCr_CL  =  4,
        CM_YFBFRV1   =  5, // Entries 5-8 correspond to FastVDO's proposed color spaces
        CM_YFBFRV2   =  6,
        CM_YFBFRV3   =  7,
        CM_YFBFRV4   =  8,
        CM_YUpVp     =  9,
        CM_ICtCp     = 10,
        CM_TOTAL
    } ColorSpace;
    """
    if v.chroma_format == ChromaFormat.YUV:
        return 0
    elif v.chroma_format == ChromaFormat.RGB:
        return 1
    raise Exception('Unknown video sequence chroma format')
    
def __chroma_format(v:VideoSequence):
    """
    typedef enum {
        CF_UNKNOWN = -1,     //!< Unknown color format
        CF_400     =  0,     //!< Monochrome
        CF_420     =  1,     //!< 4:2:0
        CF_422     =  2,     //!< 4:2:2
        CF_444     =  3,     //!< 4:4:4
        CF_TOTAL
    } ChromaFormat;
    """ 
    if v.chroma_subsampling == ChromaSubsampling.CS_420:
        return 1
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        return 2
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        return 3
    elif v.chroma_subsampling == ChromaSubsampling.CS_400:
        return 0
    raise Exception('Unknown video sequence chroma subsampling')


def __sample_range(v:VideoSequence):
    """
    typedef enum {                        // Sample range
        SR_UNKNOWN    = -1,
        SR_STANDARD   =  0,                 // Standard/Limited/Video range (16-235) * k, 
                                            // i.e. Floor (219 * D * N + 16 * D + 0.5), with N = [0, 1]
        SR_FULL       =  1,                 // Full range (0 - 2^(k-1))
        SR_RESTRICTED =  2,                 // Restricted range. Similar to SDI but less restrictive
                                            // range (4 * D .. (1 << b) - 4 * D - 1) with D = 1<<(b-10)
        SR_SDI_SCALED =  3,                 // SDI range as currently used by Dolby's PQ TIFF File format. 
                                            // Floor(1015 * D * N + 4 * D + 0.5) with N = [0, 1] and D = 1<<(b-10). 
                                            // In this mode, values are always scaled to 16 bits.
        SR_SDI        =  4,                 // SDI range as currently used by Dolby's PQ TIFF File format. 
                                            // Floor(1015 * D * N + 4 * D + 0.5) with N = [0, 1] and D = 1<<(b-10)
        SR_FZERO      =  5,                 // Full range but with zero also supported for chroma
        SR_TOTAL
    } SampleRange;
    """
    return 1 if v.video_full_range else 0


def __pixel_format(v:VideoSequence):
    """
    typedef enum {
        PF_UNKNOWN = -1,     //!< Unknown color ordering
        PF_UYVY    =  0,     //!< UYVY
        PF_YUY2    =  1,     //!< YUY2
        PF_YUYV    =  2,     //!< YUYV
        PF_YVYU    =  3,     //!< YVYU
        PF_BGR     =  4,     //!< BGR
        PF_RGB     =  5,     //!< RGB
        PF_V210    =  6,     //!< V210
        PF_UYVY10  =  7,     //!< UYVY10
        PF_V410    =  8,     //!< V410
        PF_R210    =  9,     //!< R210
        PF_R10K    =  10,    //!< R10K
        PF_XYZ     =  11,    //!< XYZ
        PF_B64A    =  12,    //!< b64a
        PF_BRG     =  13,     //!< BRG
        PF_GBR     =  14,     //!< GBR
        PF_GRB     =  15,     //!< GRB
        PF_RBG     =  16,     //!< RBG
        PF_YCM     =  17,     //!< YCM/NV12
        PF_NV12    =  17,
        PF_PCK     =  18,     //!< Packed planar YUV or RGB format for 10/12 bit data
        PF_R12B    =  19,
        PF_R12L    =  20,
        #ifdef __SIM2_SUPPORT_ENABLED__
        PF_SIM2              //!< Sim2
        #endif
        PF_TOTAL
    } PixelFormat;
    """
    # see HDRTools Input::deInterleave / Output::reInterleave
    assert not v.interleaved, "unspecified pixel format for interleaved data"
    return -1

def conversion_path(sequence:Path, suffix:str) -> Path:
    return sequence.parent / f'{sequence.stem}.conv-{suffix}.yuv'

def as_10bit_sequence(yuv_in:VideoSequence) -> VideoSequence :
    # return a modified VideoSequence obj pointing to the 10bit conversion
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 10
    yuv_out.path = conversion_path(yuv_out.path, '10bit')
    return yuv_out

def as_8bit_sequence(yuv_in:VideoSequence) -> VideoSequence :
    # return a modified VideoSequence obj pointing to the 10bit conversion
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
        '-p', f'SourceBitDepthCmp0={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp1={yuv_in.bit_depth}',
        '-p', f'SourceBitDepthCmp2={yuv_in.bit_depth}',
        '-p', f'SourceColorSpace={__color_space(yuv_in)}',
        '-p', f'SourceColorPrimaries={__color_primaries(yuv_in)}',
        '-p', f'SourceSampleRange={__sample_range(yuv_in)}', # (0: Standard, 1: Full, 2: SDI) SR_STANDARD(16-235)*k is default
        '-p', f'SourceInterleaved={int(yuv_in.interleaved)}',
        '-p', f'SourceInterlaced={int(yuv_in.interlaced)}',
        '-p', f'SourceRate={yuv_in.frame_rate}',

        # Output. Note: changing dimensions isn't implemented 
        '-p', f'OutputChromaFormat={__chroma_format(yuv_out)}',
        '-p', f'OutputBitDepthCmp0={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp1={yuv_out.bit_depth}',
        '-p', f'OutputBitDepthCmp2={yuv_out.bit_depth}',
        '-p', f'OutputRate={yuv_out.frame_rate}',

        '-p', f'NumberOfFrames={yuv_in.frame_count}', # frames to process, -1 auto based on input file
        '-p', f'StartFrame={yuv_in.start_frame - 1}',
        '-p', f'InputFileHeader=0', # header bytes count
        '-p', f'FrameSkip=0', # seems redundant with StartFrame, unclear how HDRtools uses it.

        # explicit defaults ...
        '-p', f'ChromaDownsampleFilter=2',
        '-p', f'ChromaUpsampleFilter=1',
        '-p', f'SetOutputSinglePrec=0'
    ]

    if logfile:
        opts += ['-p', f'LogFile={logfile}']

    return opts

def convert(yuv_in:VideoSequence, yuv_out:VideoSequence=None, quiet=False, dry_run=False, logfile=None):

    hdrtools_chomaconvert = os.getenv("HDRTOOLS_CHOMACONVERT", None)
    assert hdrtools_chomaconvert, "missing 'HDRTOOLS_CHOMACONVERT' env varirable path to executable"

    cmd = hdr_convert_cmd(yuv_in, yuv_out, quiet=quiet, logfile=None)
    logfile = yuv_out.path.with_suffix('.chromaconvert.log')
    run_process(logfile, hdrtools_chomaconvert, *cmd, dry_run=dry_run)

def main():
    ctx = AnchorTupleCtx.parse_args()
    assert ctx.scenario_dir.exists(), f'Not found: {ctx.scenario_dir}'
    assert len(ctx.scenario_dir.parts) >= 2, f'invalid scenario directory: {ctx.scenario_dir}'
    scenario = ctx.scenario_dir.parts[-2]
    codec = ctx.scenario_dir.parts[-1]
    base_dir = ctx.scenario_dir.parent.parent.parent
    
    executor = ProcessPoolExecutor(max_workers=3)
    
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
                    print(f'[invalid conversion] {yuv_out.path}')
                    fp.unlink()
                    yuv_out.path.unlink()
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

    # TODO: convert reference sequence and variants to RGB444
    executor.shutdown(wait=True)

    for fut in futures:
        _ = fut.result()


if __name__ == "__main__":
    main()