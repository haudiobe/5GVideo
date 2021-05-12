#!/usr/bin/python3
import argparse
import sys
import os
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Iterable, Dict, Tuple
import math

from utils import VideoSequence
from anchor import AnchorTuple, VariantData, md5_checksum, BitstreamNotFound, MetricsNotFound, ReconstructionMeta
from encoders import get_encoder
from metrics import Metric, VariantMetricSet, hdrtools_metrics, vmaf_metrics, bd_q
from enum import Enum

import pytz
utc=pytz.UTC

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


def save_verification_report(vf:Path, vd:VariantData, verification_type:AnchorVerification, success:bool, log_data:str=None, contact:str=None):

    if not success:
        assert (log_data != None) and (log_data != '')

    if vd.verification == None:
        vd.verification = { "Reports": [] }
    elif not "Reports" in vd.verification:
        vd.verification["Reports"] = []
    
    ts = datetime.now(timezone.utc)
    report = {
        "date": ts.strftime("%d/%m/%y"), # .isoformat(),
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
    
    report[key] = success

    vd.verification["Reports"].append(report)
    vd.save_as(vf)


def clear_logs(vf:Path, vd:VariantData, keep_before:datetime):

    preserved = []

    for r in vd.verification["Reports"]:
        fmt = '%d/%m/%y %H:%M:%S'
        dtstr = r['date']
        dt = datetime.strptime(dtstr, fmt)
        dt = dt.replace(tzinfo=utc)
        if dt > keep_before:
            if 'report' in r:
                assert ( vf.parent / r['report'] ).exists()
                Path.unlink( vf.parent / r['report'] )
        else:
            preserved.append(r)

    vd.verification["Reports"] = preserved
    vd.save_as(vf)

    # for f in vf.parent.iterdir():
    #     if (f.suffix == '.log') and ('verif' in f.stem):
    #         Path.unlink( f )


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('bitstream', action='store_true', default=False, help='do bitstream verification, instead of decoder/metrics verification')
    parser.add_argument('-s','--sequences', required=True, type=str, help='sequences.csv file containing the list of reference raw sequences')
    parser.add_argument('--s_dir', required=True, type=str, help='the directory containing the reference sequences')
    parser.add_argument('-a', '--anchors', required=True, type=str, help='anchors.csv file containing the list of anchors for a scenario')
    parser.add_argument('--a_dir', required=False, type=str, help='a scenario directory, containing the anchors')
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

    refs = reference_sequences_by_keys(references_csv, references_dir)

    keep_before = datetime.now(timezone.utc) - timedelta(days=2)

    for a in iter_anchors(anchors_csv, refs, anchors_dir):
        for vf, vd in iter_variants(a):
            clear_logs(vf, vd, keep_before)

if __name__ == "__main__":
    main()
