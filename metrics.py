#!/usr/bin/python3
from pathlib import Path
import struct
import io
import os
import json

from utils import VideoSequence, ColourPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction

from anchor import AnchorTuple, VariantData
from utils import run_process
from enum import Enum

class Metric(Enum):
    PSNR = "PSNR"
    PSNR_Y = "PSNR-Y"
    SSIM = "SSIM"
    MSSSIM = "MSSSIM"
    VMAF = "VMAF"
    BITRATE = "Bitrate"
    BITRATELOG = "BitrateLog"
    ENCODETIME = "EncodeTime"
    DECODETIME = "DecodeTime"


class VariantMetricSet:

    required = [
        Metric.PSNR, 
        Metric.PSNR_Y,
        Metric.MSSSIM, 
        Metric.VMAF,
        Metric.BITRATE
    ]

    encoder_stats = [
        Metric.BITRATELOG,
        Metric.ENCODETIME,
        Metric.DECODETIME
    ]
    
    def __init__(self, variant_id:str, metrics:dict):
        self.variant_id = variant_id
        for m in self.required:
            assert m in metrics, f'missing required metric {m}'
        for m in self.encoder_stats:
            if not (m in metrics):
                print('missing encoder stat: ', m)
        self.metrics = metrics

    @classmethod
    def get_keys(cls):
        return [
            "Key",
            *cls.required,
            *cls.encoder_stats
        ]

    def to_dict(self):
        filtered = { m: self.metrics[m] for m in self.required }
        for m in self.encoder_stats:
            if m in self.metrics:
                filtered[m] = self.metrics[m] 
        return {
            "Key": self.variant_id,
            **filtered
        }

################################################################################

def iter_section(f:io.FileIO, esc="-", eof_raises=True):
    l = f.readline()
    while len(l) > 0:
        if l[0] == esc:
            return
        yield l
        l = f.readline()
    assert not eof_raises, 'parser error: unexpected end of file'

def read_meta(f):
    l = f.readline()
    # prints out command line options
    meta = [*iter_section(f, esc="=")]
    # prints out input sequences' descriptions
    meta += [*iter_section(f, esc="=")]
    l = f.readline()
    assert len(l) > 0 and l[0] == '-', 'failed to parse header'
    return meta

def read_log(f, stats={}):
    stats['meta'] = read_meta(f)
    stats['metrics'] = [l.split() for l in iter_section(f)][0]
    stats['frames'] = [l.split() for l in iter_section(f)]
    stats['avg'] = [l.split() for l in iter_section(f)][0]
    amplitude = [l.split() for l in iter_section(f)]
    stats['min'] = amplitude[0]
    stats['max'] = amplitude[1]
    stats['perfs'] = "".join([*iter_section(f)])
    return stats

def parse_metrics(log):
    raw_data = None
    p = Path(log).resolve()
    with open(p, 'r') as f:
        raw_data = read_log(f)
    assert raw_data != None, f'failed to parse {p}'
    parse_hex = lambda indices, values: [struct.unpack('!d', bytes.fromhex(values[i]))[0] for i in indices]
    parse_frames = lambda indices, frames: [[f[0], *parse_hex(indices, f)] for f in frames]
    indices, metrics = zip(*[(i, h[3:]) for i, h in enumerate(raw_data['metrics']) if str(h).startswith('hex')])
    return {
        'metrics': metrics,
        'avg': parse_hex(indices, raw_data['avg']),
        'min': parse_hex(indices, raw_data['min']),
        'max': parse_hex(indices, raw_data['max']),
        'frames': [['frame', *metrics], *parse_frames(indices, raw_data['frames'])]
    }

def hdrtools_input(v:VideoSequence, ref=True, start_frame=0, file_header=0):
    i = 0 if ref else 1
    opts = [
        '-p', f'Input{i}File={v.path}',
        '-p', f'Input{i}Width={v.width}',
        '-p', f'Input{i}Height={v.height}',
        '-p', f'Input{i}BitDepthCmp0={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp1={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp2={v.bit_depth}',
        '-p', f'Input{i}StartFrame={start_frame}',
        '-p', f'Input{i}FileHeader={file_header}',
        '-p', f'Input{i}Rate={v.frame_rate}',
        '-p', f'Input{i}SampleRange=0', # SR_STANDARD is HDRMetrics' default, (16-235)*k
        # '-p', f'Input{i}FourCCCode={0}' # PF_UYVY is HDRMetrics' default, specifies custom pixel formats, mostly for interleaved and custom component ordering (eg. BGR instead of RGB)
    ]
    
    if v.interleaved:
        opts += ['-p', f'Input{i}Interleaved=0']
    else:
        opts += ['-p', f'Input{i}Interleaved=1']

    if v.interlaced:
        opts += ['-p', f'Input{i}Interlaced=0']
    else:
        opts += ['-p', f'Input{i}Interlaced=1']

    if v.chroma_format == ChromaFormat.YUV:
        opts += ['-p', f'Input{i}ColorSpace=0'] # 0:CM_YCbCr
    elif v.chroma_format == ChromaFormat.RGB:
        opts += ['-p', f'Input{i}ColorSpace=1'] # 1:CM_RGB
    else:
        # out of scope
        raise ValueError('Unexpected color space')
    
    if v.chroma_subsampling == ChromaSubsampling.CS_400:
        opts += ['-p', f'Input{i}ChromaFormat=0']
    elif v.chroma_subsampling == ChromaSubsampling.CS_420:
        opts += ['-p', f'Input{i}ChromaFormat=1']
    elif v.chroma_subsampling == ChromaSubsampling.CS_422:
        opts += ['-p', f'Input{i}ChromaFormat=2']
    elif v.chroma_subsampling == ChromaSubsampling.CS_444:
        opts += ['-p', f'Input{i}ChromaFormat=3']

    if v.colour_primaries == ColourPrimaries.BT_709:
        opts += ['-p', f'Input{i}ColorPrimaries=0']
    elif v.colour_primaries == ColourPrimaries.BT_2020:
        opts += ['-p', f'Input{i}ColorPrimaries=1']

    return opts


def hdrtools_metrics(a:AnchorTuple, reconstructed:Path) -> dict:
    """
    the function assumes that reconstructed and reference sequences have the same properties
        eg. size, bitdepth, chroma ...
    """
    # PQ content metrics to be parsed from VTM encoder log
    assert a.reference.transfer_characteristics != TransferFunction.BT2020_PQ, 'unsupported transfer function'

    ref = a.reference
    dist = VideoSequence(reconstructed, **ref.properties)

    input0 = hdrtools_input(ref, ref=True, start_frame=a.start_frame)
    input1 = hdrtools_input(dist, ref=False, start_frame=0)
    run = [
        '-p', f'EnablehexMetric=1', # the parser collects hex values only
        '-p', f'EnableJVETPSNR=1',
        '-p', f'EnableShowMSE=1',
        '-p', f'EnablePSNR=1',
        '-p', f'EnableSSIM=1',
        '-p', f'EnableMSSSIM=1',
        '-p', f'SilentMode=0',
        '-p', f'NumberOfFrames={a.frame_count}'
    ]

    distortion_log = reconstructed.with_suffix('.distortion.txt')
    run += ['-p', f'LogFile={distortion_log}']

    tool = os.getenv('HDRMETRICS_TOOL')
    cmd = [tool, *input0, *input1, *run]

    log = reconstructed.with_suffix('.metrics.log')
    run_process(log, *cmd, dry_run=a.dry_run)
    if a.dry_run:
        return {}
    
    data = parse_metrics(log)
    metrics = { k:v for k,v in zip(data['metrics'], data['avg']) }
    
    if ref.chroma_format == ChromaFormat.YUV:
        return {
            Metric.PSNR: ((6 * metrics["PSNR-Y"]) + metrics["PSNR-U"] + metrics["PSNR-V"]) / 8,
            Metric.SSIM: ((6 * metrics["SSIM-Y"]) + metrics["SSIM-U"] + metrics["SSIM-V"]) / 8,
            Metric.MSSSIM: ((6 * metrics["MSSSIM-Y"]) + metrics["MSSSIM-U"] + metrics["MSSSIM-V"]) / 8,
            **metrics
        }
    elif ref.chroma_format == ChromaFormat.RGB:
        return {
            Metric.PSNR: ( metrics["PSNR-R"] + metrics["PSNR-G"] + metrics["PSNR-B"] ) / 3,
            Metric.SSIM: ( metrics["SSIM-R"] + metrics["SSIM-G"] + metrics["SSIM-B"] ) / 3,
            Metric.MSSSIM: ( metrics["MSSSIM-R"] + metrics["MSSSIM-G"] + metrics["MSSSIM-B"] ) / 3,
            **metrics
        }


################################################################################

def vmaf_metrics(a:AnchorTuple, reconstructed:Path, model="version=vmaf_v0.6.1"):
    output = reconstructed.with_suffix('.vmaf.json')
    log = reconstructed.with_suffix('.vmaf.log')
    vmaf_exec = os.getenv('VMAF_EXEC', 'vmaf')
    cmd = [
        vmaf_exec,
        "-r", f'{a.reference.path}', 
        "-d", f'{reconstructed}',
        "-w", f'{a.reference.width}',
        "-h", f'{a.reference.height}',
        "-p", f'{a.reference.chroma_subsampling.value}',
        "-b", f'{a.reference.bit_depth}',
        "--json", "-o", str(output),
        "-m", model
    ]
    run_process(log, *cmd, dry_run=a.dry_run)
    with open(output, "rb") as fp:
        data = json.load(fp)
        return data["pooled_metrics"]["vmaf"]["mean"]


################################################################################


def bd_q(RA, QA, RT, QT, piecewise=0):
    import numpy as np
    import scipy
    
    lRA = np.log(RA)
    lRT = np.log(RT)
    QA = np.array(QA)
    QT = np.array(QT)
    p1 = np.polyfit(lRA, QA, 3)
    p2 = np.polyfit(lRT, QT, 3)

    # integration interval
    min_int = max(min(lRA), min(lRT))
    max_int = min(max(lRA), max(lRT))
    
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lRA), QA[np.argsort(lRA)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lRT), QT[np.argsort(lRT)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)
    return avg_diff

