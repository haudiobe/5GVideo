#!/usr/bin/python3
from pathlib import Path
import struct
import io
import os
import json
import csv
import math
import functools
import copy
import shlex

import numpy as np
from numpy.polynomial import Polynomial, polynomial as P
import scipy.interpolate

from utils import VideoSequence, ColorPrimaries, ChromaFormat, ChromaSubsampling, TransferFunction

from anchor import AnchorTuple, VariantData, Metric, VariantMetricSet, iter_variants
from utils import run_process
from encoders import get_encoder, parse_encoding_bitdepth, parse_variant_qp
from convert import as_10bit_sequence, as_8bit_sequence
from enum import Enum

from typing import List, Dict, Generator, Iterable

class VideoFormatException(BaseException):
    pass

SDR_METRICS = (
        Metric.BITRATELOG,
        Metric.BITRATE,
        Metric.PSNR_Y,
        Metric.PSNR_U,
        Metric.PSNR_V,
        Metric.PSNR,
        Metric.MSSSIM, 
        Metric.VMAF,
        Metric.ENCODETIME,
        Metric.DECODETIME
)


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
    # stats['perfs'] = "".join([*iter_section(f)])
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

def hdrtools_input(v:VideoSequence, ref=True, file_header=0):
    i = 0 if ref else 1
    opts = [
        '-p', f'Input{i}File={v.path}',
        '-p', f'Input{i}Width={v.width}',
        '-p', f'Input{i}Height={v.height}',
        '-p', f'Input{i}BitDepthCmp0={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp1={v.bit_depth}',
        '-p', f'Input{i}BitDepthCmp2={v.bit_depth}',
        '-p', f'Input{i}StartFrame={v.start_frame-1}',
        '-p', f'Input{i}FileHeader={file_header}',
        '-p', f'Input{i}Rate={v.frame_rate}',
        '-p', f'Input{i}SampleRange=0', # SR_STANDARD is HDRMetrics' default, (16-235)*k
        # '-p', f'Input{i}FourCCCode={0}' # PF_UYVY is HDRMetrics' default, specifies custom pixel formats, mostly for interleaved and custom component ordering (eg. BGR instead of RGB)
    ]
    
    if v.interleaved:
        opts += ['-p', f'Input{i}Interleaved=1']
    else:
        opts += ['-p', f'Input{i}Interleaved=0']

    if v.interlaced:
        opts += ['-p', f'Input{i}Interlaced=1']
    else:
        opts += ['-p', f'Input{i}Interlaced=0']

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

    if v.colour_primaries == ColorPrimaries.BT_709:
        opts += ['-p', f'Input{i}ColorPrimaries=0']
    elif v.colour_primaries == ColorPrimaries.BT_2020:
        opts += ['-p', f'Input{i}ColorPrimaries=1']

    return opts


def hdrtools_metrics(ref:VideoSequence, dist:VideoSequence, dry_run=False, cfg:Path=None) -> dict:

    # PQ content metrics to be parsed from VTM encoder log
    assert ref.transfer_characteristics != TransferFunction.BT2020_PQ, f'unsupported transfer function, {TransferFunction.BT2020_PQ}'
    assert ref.chroma_format == ChromaFormat.YUV, f'unsupported chroma format {ChromaFormat.YUV}'
    
    input0 = hdrtools_input(ref, ref=True)
    input1 = hdrtools_input(dist, ref=False)
    run = ['-f', str(cfg)] if cfg else []
    run += [
        '-p', f'EnablehexMetric=1', # the parser collects hex values only
        '-p', f'EnableJVETPSNR=1',
        '-p', f'EnableShowMSE=1',
        '-p', f'EnablePSNR=1',
        # '-p', f'EnableSSIM=1',
        # '-p', f'EnableMSSSIM=1',
        '-p', f'SilentMode=0',
        '-p', f'NumberOfFrames={dist.frame_count}'
    ]

    distortion_log = dist.path.with_suffix('.distortion.txt')
    run += ['-p', f'LogFile={distortion_log}']

    tool = os.getenv('HDRMETRICS_TOOL')
    cmd = [tool, *input0, *input1, *run]

    log = dist.path.with_suffix('.metrics.log')
    run_process(log, *cmd, dry_run=dry_run)
    if dry_run:
        return {
            Metric.PSNR_Y.key: -1,
            Metric.PSNR_U.key: -1,
            Metric.PSNR_V.key: -1,
            Metric.MSSSIM.key: -1
        }

    data = parse_metrics(log)
    metrics = { k:v for k,v in zip(data['metrics'], data['avg']) }

    if ref.chroma_format == ChromaFormat.YUV:
        (y,u,v) = (metrics["PSNR-Y"], metrics["PSNR-U"], metrics["PSNR-V"])
        log_msssim = metrics["JMSSSIM-Y"]
        return {
            Metric.PSNR_Y.key: y,
            Metric.PSNR_U.key: u,
            Metric.PSNR_V.key: v,
            Metric.MSSSIM.key: log_msssim
        }
    else:
        return metrics

################################################################################

def bitstream_size(bitstream:Path, drop_sei=False) -> int:
    tmp = None
    if drop_sei:
        # expecting SEIRemovalAppStatic built from HM16.23
        tool = os.getenv('SEI_REMOVAL_APP')
        assert tool, 'missing env variable: SEI_REMOVAL_APP'
        tmp = bitstream.with_suffix('.tmp')
        cmd = [ tool, '-b', str(bitstream), '-o', str(tmp), f'--DiscardPrefixSEI=1', f'--DiscardSuffixSEI=1' ]
        log = bitstream.with_suffix('.seiremoval.log')
        run_process(log, *cmd, dry_run=False)
    s = int(os.path.getsize(tmp if tmp else bitstream))
    if tmp:
        os.remove(tmp)
        os.remove(log)
    return s

################################################################################

def vmaf_metrics(ref:VideoSequence, dist:VideoSequence, model="version=vmaf_v0.6.1", dry_run=False):
    # VMAF may not support skipping frames at the beginning of input sequences
    output = dist.path.with_suffix('.vmaf.json')
    log = dist.path.with_suffix('.vmaf.log')
    vmaf_exec = os.getenv('VMAF_EXEC', 'vmafossexec')
    """
    cmd = [
        vmaf_exec,
        "-r", f'{ref.path}', 
        "-d", f'{dist.path}',
        "-w", f'{ref.width}',
        "-h", f'{ref.height}',
        "-p", f'{ref.chroma_subsampling.value}',
        "-b", f'{ref.bit_depth}',
        "--json", "-o", str(output),
        "-m", model
    ]
    """
    print(f'[{ref.bit_depth}\tbits]::', ref.path.name)
    print(f'[{dist.bit_depth}\tbits]::', dist.path.name)
    try:
        assert not dist.interlaced and not ref.interlaced
        assert not dist.interleaved and not ref.interleaved
        # dist.chroma_format
        # dist.chroma_subsampling
    except AssertionError:
        print('Invalid format for VMAF. Must be one of yuv420p, yuv420p10le, yuv420p12le, yuv420p16le')

    pixfmt = None 
    if ref.bit_depth == 8:
        pixfmt = 'yuv420p'
    elif ref.bit_depth == 10:
        pixfmt = 'yuv420p10le'
    elif ref.bit_depth == 12:
        pixfmt = 'yuv420p12le'
    elif ref.bit_depth == 16:
        pixfmt = 'yuv420p16le'

    cmd = [
        vmaf_exec,
        str(pixfmt),
        str(ref.width),
        str(ref.height),
        str(ref.path),
        str(dist.path),
        str(model),
        '--log', str(output),
        '--log-fmt', 'json'
    ]
    
    run_process(log, *cmd, dry_run=dry_run)
    if output.exists():
        with open(output, "rb") as fp:
            data = json.load(fp)
            return data["pooled_metrics"]["vmaf"]["mean"]
    else:
        print(f"VMAF output not found : {output}")
        return None


################################################################################

def compute_metrics(a:AnchorTuple, vd:VariantData, vmaf=True, encoder_log=True, decoder_log=False, preprocessed=True, dist_dir:Path=None) -> VariantMetricSet:
    print('compute_metrics()')
    # check if a pre-conversion step is needed, 
    coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
    if dist_dir == None:
        dist_dir = a.working_dir
    if not preprocessed:
        ref = a.reference
        dist = VideoSequence.load(dist_dir / f'{vd.variant_id}.yuv.json')
    elif (coded_bit_depth == a.reference.bit_depth):
        ref = a.reference
        dist = VideoSequence.load(dist_dir / f'{vd.variant_id}.yuv.json')
    elif (a.reference.bit_depth == 8) and (coded_bit_depth == 10):
        # use a 10 bit conversion instead
        ref = as_10bit_sequence(a.reference)
        assert ref.path.exists(), f'reference sequence needs pre-processing - Not found: {ref.path}'
        dist = VideoSequence.from_sidecar_metadata(dist_dir / f'{vd.variant_id}.yuv.json')
        assert ref.bit_depth == dist.bit_depth

    print('ready for metrics')

    if not os.getenv('DISABLE_HDRMETRICS'):
        metrics_cfg = os.getenv('HDRMETRICS_CFG')
        d = hdrtools_metrics(ref, dist, dry_run=a.dry_run, cfg=Path(metrics_cfg) if metrics_cfg else None)
        metrics = VariantMetricSet(d)
        metrics.compute_avg_psnr(strict=False)
    else:
        metrics = VariantMetricSet()
        metrics[Metric.PSNR_Y.key] = None
        metrics[Metric.PSNR_U.key] = None
        metrics[Metric.PSNR_V.key] = None
        metrics[Metric.PSNR.key]   = None
        metrics[Metric.MSSSIM.key] = None
        
    if os.getenv('DISABLE_VMAF'):
        metrics[Metric.VMAF.key] = 0
    else:
        mdl = os.getenv('VMAF_MODEL', "version=vmaf_v0.6.1")
        metrics[Metric.VMAF.key] = vmaf_metrics(ref, dist, mdl, dry_run=a.dry_run)
    
    if a.dry_run:
        return vd.variant_id, VariantMetricSet()

    bitstream = a.working_dir / vd.bitstream['URI']

    s = bitstream_size(bitstream, drop_sei=False)
    metrics[Metric.BITRATE.key] = int(s * 8 / a.duration) * 1e-3

    enc = get_encoder(vd.generation['encoder'])

    # parse additional metrics from ENCODER log 
    if 'log-file' in vd.generation:
        enc_log = a.working_dir / vd.generation['log-file']
        if enc_log.exists():
            encoder_metrics = enc.encoder_log_metrics(enc_log)
            if Metric.BITRATELOG.key in encoder_metrics:
                metrics[Metric.BITRATELOG.key] = float(encoder_metrics[Metric.BITRATELOG.key])
            if Metric.ENCODETIME.key in encoder_metrics:
                metrics[Metric.ENCODETIME.key] = float(encoder_metrics[Metric.ENCODETIME.key])
        else:
            print(f'#\tencoder log not found: {enc_log}')
            metrics[Metric.BITRATELOG.key] = 0
            metrics[Metric.ENCODETIME.key] = 0
    
    # parse additional metrics from DECODER log 
    if vd.reconstruction and vd.reconstruction.get('log-file', None):
        dec_log = a.working_dir / vd.reconstruction['log-file']
        if dec_log.exists():
            decoder_metrics = enc.decoder_log_metrics(dec_log)
            if Metric.DECODETIME.key in decoder_metrics:
                metrics[Metric.DECODETIME.key] = float(decoder_metrics[Metric.DECODETIME.key])
        else:
            print(f'#\tdecoder log not found: {dec_log}')
            metrics[Metric.DECODETIME.key] = 0

    return metrics


def anchor_metrics_to_csv(a:AnchorTuple, dst:Path=None):
    fieldnames = None
    for variant_path, variant_data in iter_variants(a):
        assert variant_path.exists(), f'{variant_path} not found'
        assert variant_data.metrics, f'metrics not defined in: {variant_path}'
        if not fieldnames:
            fieldnames = ["parameter", *variant_data.metrics.keys()]

    if dst == None:
        dst = a.working_dir.parent / 'Metrics' / f'{a.working_dir.stem}.csv'
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    with open(dst, 'w') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for variant_path, variant_data in iter_variants(a):
            row = {}
            for k in fieldnames:
                if k == "parameter":
                    row[k] = parse_variant_qp(variant_data.variant_cli)
                else:
                    row[k] = variant_data.metrics[k]
            writer.writerow(row)


################################################################################

def compute_avg_psnr(vd:VariantData):
    try:
        if 'psnr' not in vd.metrics:
            [Y, U, V] = [vd.metrics[k.key] for k in [Metric.PSNR_Y, Metric.PSNR_U, Metric.PSNR_V]]
        vd.metrics[Metric.PSNR.key] = ((6*Y)+U+V)/8
    except BaseException as e:
        vd.metrics[Metric.PSNR.key] = str(e)

def compute_log_msssim(vd:VariantData):
    try:
        inv = 1 - vd.metrics[Metric.MSSSIM.key]
        vd.metrics[Metric.MSSSIM.key] = -math.inf if inv == 0 else -10.0 * math.log10(inv)
    except BaseException as e:
        vd.metrics[Metric.MSSSIM.key] = str(e)

################################################################################

def sort_rates_on(rates, metric):
    """
    sort [(rate, psnr), ...] samples based on psnr 
    """
    return rates[np.argsort(metric)], np.sort(metric)

def sort_on_rates(rates, metric):
    """
    sort [(rate, psnr), ...] samples based on rate 
    """
    return np.sort(rates), metric[np.argsort(rates)]


def sanitize_rd_data2(rates, PSNR, step=0.001):
    """- sort samples for increasing rates
    - fix saturated values, by adding a step value"""
    rate = np.array(rates)
    dist = np.array(PSNR)
    sorted = np.lexsort((rate, dist))
    rate = rate[sorted]
    dist = dist[sorted]
    dist_max = dist[-1]
    dist_fix = np.array([], dtype=np.float64)
    step = 0.001
    fix = 0.001
    for i, _ in enumerate(rate):
        d = dist[i]
        if d == dist_max:
            d += fix
            fix += step
        dist_fix.append(d)
    return rate, dist_fix


def sanitize_rd_data1(rates, PSNR, preserve_last_sample=True):
    """- sort samples for increasing rates
    - drop samples on non increasing dist values"""
    rate = np.array(rates)
    dist = np.array(PSNR)
    sorted = np.lexsort((rate, dist))
    rate = rate[sorted]
    dist = dist[sorted]
    _sorted = []
    for i, r in enumerate(rate):
        d = dist[i]
        if len(_sorted):
            p = _sorted[-1]
            if (d <= p[1]):
                print(f'/!\ rate increased, but quality did not - dropping sample ( r:{r}, d:{d} ) !')
                # keep the last sample on saturated values, drop the previous one
                # if preserve_last_sample and (len(rate) == (i+1)):
                #     dist.pop()
                # else:
                #     continue
                continue
        _sorted.append((r, d))
    return [np.array(arr) for arr in zip(*_sorted)]



def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=1, sanitize=False) -> float:

    if sanitize:
        R1, PSNR1 = sanitize_rd_data1(R1, PSNR1)
        R2, PSNR2 = sanitize_rd_data1(R2, PSNR2)
    else:
        PSNR1 = np.array(PSNR1)
        PSNR2 = np.array(PSNR2)

    """
    adapted from https://github.com/Anserw/Bjontegaard_metric
    which computes bd-rate according to:
        [1] G. Bjontegaard, Calculation of average PSNR differences between RD-curves (VCEG-M33) 
        [2] S. Pateux, J. Jung, An excel add-in for computing Bjontegaard metric and its evolution
    """

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    if piecewise == 0:
        p1 = Polynomial.fit(PSNR1, lR1, 3)
        p2 = Polynomial.fit(PSNR2, lR2, 3)
        
        p_int1 = P.polyint(p1)
        p_int2 = P.polyint(p2)
        
        int1 = P.polyval(p_int1, max_int) - P.polyval(p_int1, min_int)
        int2 = P.polyval(p_int2, max_int) - P.polyval(p_int2, min_int)

    else:
        samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
        [y1, x1] = sort_on_rates(lR1, PSNR1)
        [y2, x2] = sort_on_rates(lR2, PSNR2)

        err = None
        try:
            v1 = scipy.interpolate.pchip_interpolate(x1, y1, samples)
            v2 = scipy.interpolate.pchip_interpolate(x2, y2, samples)

            # Calculate the integral using the trapezoid method on the samples.
            int1 = np.trapz(v1, dx=interval)
            int2 = np.trapz(v2, dx=interval)

        except BaseException as e:
            err = e

    if err:
        print(err)
        avg_diff = 0
    else:
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1) * -100
    
    return avg_diff, R1, PSNR1, R2, PSNR2
