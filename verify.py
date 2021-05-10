#!/usr/bin/python3
import argparse
import sys
import os
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone 
from typing import List, Iterable, Dict, Tuple
import math

from utils import VideoSequence
from anchor import AnchorTuple, VariantData, md5_checksum, BitstreamNotFound, MetricsNotFound, ReconstructionMeta
from encoders import get_encoder
from metrics import Metric, VariantMetricSet, hdrtools_metrics, vmaf_metrics, bd_q
from enum import Enum

ENCODING = 'utf-8-sig'

# anchors/scenario/anchors.csv
class AnchorList:
    KEY = '#Key' # directory where anchor is stored
    CLAUSE = 'Clause'
    REF_SEQ = 'Reference Sequence'
    REF_ENC = 'Reference_Encoder'
    CFG = 'Configuration'
    VARIANTS = 'Variations'
    VARIANT_KEY = 'Anchor_Key' # template for variant json filename

# anchors/scenario/sequences.csv
class RefSequenceList:
    KEY = '#Key'
    NAME = 'Name'
    REF = 'Reference'
    LOC = 'Location'
    DUR = 'Duration'

class AnchorVerification(Enum):
    BITSTREAM = 1
    DECODER = 2

def reference_sequences_by_keys(reference_list:Path, root_dir:Path=Path('.')) -> Dict[str, VideoSequence]:
    refs = {}
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            loc = row[RefSequenceList.LOC]
            meta = root_dir / loc / f'{loc}.json'
            vs = VideoSequence.from_sidecar_metadata(meta)
            # assert vs.start_frame == 0 // starts from 1 ???
            assert (vs.frame_count / vs.frame_rate) == float(row[RefSequenceList.DUR]), f'(frame_count / frame_rate) != expected duration'
            refs[row[RefSequenceList.KEY]] = vs
    return refs

def iter_anchors(anchor_list:Path, refs:Dict[str, VideoSequence], root_dir:Path) -> Iterable[AnchorTuple]:
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            seq = refs[row[AnchorList.REF_SEQ]]
            description = row[AnchorList.CLAUSE]
            encoder_id = row[AnchorList.REF_ENC] # eg. HM16.22, 
            encoder_cfg = row[AnchorList.CFG] # eg. S3-HM-01, no directory context specified
            variants = row[AnchorList.VARIANTS]
            anchor_dir = root_dir / row[AnchorList.KEY]
            anchor_key = row[AnchorList.VARIANT_KEY]
            yield AnchorTuple(anchor_dir, seq, encoder_id, encoder_cfg, variants, anchor_key, description, seq.start_frame, seq.frame_count)

def iter_variants(a:AnchorTuple) -> Iterable[Tuple[Path, VariantData]]:
    """ 
    yields (a.working_dir/variant.json, VariantData)
    """
    encoder = get_encoder(a.encoder_id)
    assert encoder != None, f'encoder not supported: {a.encoder_id}'

    for variant_id, _ in a.iter_variants():
        vfp = a.working_dir / f'{variant_id}.json'
        data = None
        if vfp.exists():
            data = VariantData.load(vfp)
        # vd = VariantData.load(vfp)
        # assert vd.variant_id == variant_id, f'variant id mismatch | parsed: "{vd.variant_id}" | expected: "{variant_id}"'
        yield vfp, data

def compute_metrics(a:AnchorTuple, vd:VariantData, r:ReconstructionMeta) -> VariantMetricSet:

    metrics = hdrtools_metrics(a, r.reconstructed)
    
    if a.start_frame == 0:
        # https://github.com/Netflix/vmaf/blob/master/libvmaf/tools/README.md#vmaf-models
        # 'enable_transform' (aka --phone-model)
        # see: https://github.com/Netflix/vmaf/blob/master/libvmaf/tools/cli_parse.c#L188 
        # and: https://github.com/Netflix/vmaf/blob/master/python/vmaf/script/run_vmaf.py#L80
        mdl = os.getenv('VMAF_MODEL', "version=vmaf_v0.6.1:enable_transform")
        metrics[Metric.VMAF] = vmaf_metrics(a, r, mdl) # TBD. VMAF metric should come with a model definition
    else: # TBD. VMAF itself does not support adressing a segment, using vmaf through libav/ffmpeg would easily solve this issue
        metrics[Metric.VMAF] = None

    if a.dry_run:
        return None

    bitstream = a.working_dir / vd.bitstream['URI']
    metrics[Metric.BITRATE] = int(os.path.getsize(bitstream) * 8 / a.duration)

    enc = get_encoder(vd.generation['encoder'])

    enc_log = a.working_dir / vd.generation['log-file']
    if enc_log.exists():
        encoder_metrics = enc.encoder_log_metrics(enc_log)
        metrics = { **metrics, **encoder_metrics }
    else:
        print(f'enccoder log not found: {enc_log}')

    dec_log = a.working_dir / vd.reconstruction['log-file']
    if dec_log.exists():
        decoder_metrics = enc.decoder_log_metrics(dec_log)
        metrics = { **metrics, **decoder_metrics }
    else:
        print(f'decoder log not found: {dec_log}')
    
    return VariantMetricSet(vd.variant_id, metrics)


def save_verification_report(vf:Path, vd:VariantData, verification_type:AnchorVerification, success:bool, log_data:str=None, contact:str=None):

    if not success:
        assert (log_data != None) and (log_data != '')

    if vd.verification == None:
        vd.verification = { "Reports": [] }
    elif not "Reports" in vd.verification:
        vd.verification["Reports"] = []
    
    ts = datetime.now(timezone.utc)
    report = {
        "date": ts.strftime('%d/%m/%y %H:%M:%S'), # .isoformat(),
        "contact": contact
    }

    idx = len(vd.verification["Reports"])

    if verification_type == AnchorVerification.BITSTREAM:
        key = "bistream"
    elif verification_type == AnchorVerification.DECODER:
        key = "decoder"

    if not success:
        log_file = vf.parent / f'{vd.variant_id}_verification_{idx}_{key}_{ts.strftime("%d%m%y")}.log'
        with open(log_file, 'w') as fo:
            fo.write(log_data)
        report["report"] = str(Path('.') / log_file.name)

    report[key] = success

    vd.verification["Reports"].append(report)
    vd.save_as(vf)



#########################################################################################################

def decoder_verification_preflight(a:AnchorTuple):
    """
    verifies that:
        * all variant bitstreams exists
        * all bitstreams md5 match their json description
        * has expected metric set
    """
    err = None
    for vf, vd in iter_variants(a):
        try:
            _ = vd.locate_bitstream(a.working_dir, md5_check=True)
            vd.has_metric_set( *VariantMetricSet.get_keys(a) )
        except BaseException as e:
            if err == None:
                err = []
            err.append((vf, vd, e))
    return err


def verify_variant_metrics(a:AnchorTuple, vd:VariantData, vf:Path, tmp_dir=True) -> Tuple[bool, str]:
    """
    1. decode into a temporary folder
    2. compute metrics with freshly decoded data
    3. compare with values defined in VariantData
    return a diff in case of missmatch
    """
    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    dec = get_encoder(enc_id)

    # compute metrics for verification
    tmp = vf.parent.with_suffix('.tmp')
    tmp.mkdir(exist_ok=(not tmp_dir))

    r = dec.decode_variant(a, vd, tmp)
    if not a.dry_run:
        assert r.reconstructed_md5 == vd.reconstruction['md5']

    metrics_new = compute_metrics(a, vd, r)
    if a.dry_run:
        return False, 'dry-run'

    report = {}
    for m in VariantMetricSet.get_keys(a):
        found = metrics_new[m]
        expected = vd.metrics[m]
        if math.isclose(found, expected, rel_tol=1e-4):
            report[m] = expected - found
    
    if tmp_dir:
        shutil.rmtree(tmp)
    
    if len(report) == 0:
        return True, None
    else:
        log = 'Metric key:\tDIFF\n'
        for k, v in report:
            log += f'{k}:\t{v}\n'
        return False, log


def verify_anchor_metrics(a:AnchorTuple, contact:str=None):
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        success, log_data = verify_variant_metrics(a, vd, vf, tmp_dir=False)
        save_verification_report(vf, vd, AnchorVerification.DECODER, success, log_data, contact=contact)
        if not success:
            a_errors.append(log_data)
    return a_errors


#########################################################################################################

def bitstream_verification_preflight(a:AnchorTuple):
    """
    verifies:
        * all variant bitstreams exists
        * all bitstreams md5 match their json description
    """
    err = None
    for vf, vd in iter_variants(a):
        try:
            _ = vd.locate_bitstream(a.working_dir, md5_check=True)
        except BaseException as e:
            if err == None:
                err = []
            err.append((vf, vd, e))
    return err


def verify_variant_bitstream(a:AnchorTuple, vd:VariantData, vf:Path, tmp_dir=True) -> Tuple[bool, str]:
    """
    verifies:
        * re-encoding a new bitstream matches md5 of the existing bitstream
    """
    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    enc = get_encoder(enc_id)

    tmp = vf.parent.with_suffix('.tmp')
    tmp.mkdir(exist_ok=(not tmp_dir))

    vd_new = dec.encode_variant(a, vd.variant_id, vd.variant_cli, tmp)
    md5_new = vd_new.bitstream["md5"]
    md5_ref = vd.bitstream["md5"]
    
    if tmp_dir:
        shutil.rmtree(tmp)

    if md5_new == md5_ref:
        return True
    else:
        return False, f'invalid md5 - expected:{md5_new} - found:{md5_ref}'


def verify_anchor_bitstreams(a:AnchorTuple, contact:str=None):
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        success, log_data = verify_variant_metrics(a, vd, vf, tmp_dir=False)
        save_verification_report(vf, vd, AnchorVerification.BITSTREAM, success, log_data, contact=contact)
        if not success:
            a_errors.append(log_data)
    return a_errors


#########################################################################################################

def verify(verification_type:AnchorVerification, refs:Dict[str, VideoSequence], anchors:Iterable[AnchorTuple], dry_run=True, contact=None):

    batch = []
    preflight_errors = {}

    if verification_type == AnchorVerification.DECODER:
        preflight_fn = decoder_verification_preflight
        verification_fn = verify_anchor_metrics

    elif verification_type == AnchorVerification.BITSTREAM:
        preflight_fn = bitstream_verification_preflight
        verification_fn = verify_anchor_bitstreams

    for a in anchors:
        err = preflight_fn(a)
        if err == None:
            batch.append(a)
        else:
            preflight_errors[a] = err
            print(f'{a} - error - can not run verification:\n * {err}')

    for a in batch:
        a.dry_run = dry_run
        errors = verification_fn(a, contact)
        print(f'{a} - error - verification failed:\n * {errors}')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('bitstream', help='do bitstream verification, instead of decoder/metrics verification')
    # parser.add_argument('bitstream', action='store_true', default=False, help='do bitstream verification, instead of decoder/metrics verification')
    parser.add_argument('-s','--sequences', required=True, type=str, help='sequences.csv file containing the list of reference raw sequences')
    parser.add_argument('--s_dir', required=True, type=str, help='the directory containing the reference sequences')
    parser.add_argument('-a', '--anchors', required=True, type=str, help='anchors.csv file containing the list of anchors for a scenario')
    parser.add_argument('--a_dir', required=False, type=str, help='a scenario directory, containing the anchors')
    parser.add_argument('-c', '--contact', required=True, type=str, help='email for the verification report contact field')
    parser.add_argument('-d', '--dry_run', action='store_true', default=False)
    args = parser.parse_args()

    references_csv = Path(args.sequences)
    assert references_csv.exists()
    references_dir = Path(args.s_dir)
    assert references_dir.is_dir()

    anchors_csv = Path(args.sequences)
    assert references_csv.exists()
    anchors_dir = Path(args.s_dir)
    assert references_dir.is_dir()

    anchors_csv = Path(args.anchors)
    anchors_dir = Path(args.a_dir)

    if args.bitstream:
        verification_type = AnchorVerification.BITSTREAM
    else:
        verification_type = AnchorVerification.DECODER

    refs = reference_sequences_by_keys(references_csv, references_dir)
    anchors = iter_anchors(anchors_csv, refs, anchors_dir)
    verify(
        verification_type,
        refs, 
        anchors, 
        dry_run=args.dry_run,
        contact=args.contact
    )


if __name__ == "__main__":
    main()
