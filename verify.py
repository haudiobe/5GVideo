#!/usr/bin/python3

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

def to_csv(fout:str, fieldnames:List[str], values:List[dict]):
    with open(fout, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(row)


def parse_args():

    assert len(sys.argv) == 3, 'cmd.py ./anchors.csv ./references.csv'

    anchors = Path(sys.argv[1])
    assert anchors.exists()

    references = Path(sys.argv[2])
    assert references.exists()

    # assert os.getenv('HDRMETRICS_TOOL') != None, 'HDRMETRICS_TOOL environment variable not set'
    return anchors, references

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

    dec_log = a.working_dir / vd.reconstruction['log-file']
    if dec_log.exists():
        decoder_metrics = enc.decoder_log_metrics(enc_log)
        metrics = { **metrics, **decoder_metrics }
    
    return VariantMetricSet(vd.variant_id, metrics)

def compute_variant_metrics(a:AnchorTuple, vd:VariantData):
    dec = get_encoder(a.encoder_id)
    reconstruction = dec.decode_variant(vd, a)
    metrics = compute_metrics(a, vd, reconstruction)
    return reconstruction, metrics

def verify_metrics(vf:Path, vd:VariantData, a:AnchorTuple, tmp_dir=True) -> dict:
    """
    1. decode into a temporary folder, 
    2. compute metrics on it, 
    3. compare with values defined in VariantData
    4. update VariantData, return the list of errors
    """
    # sanity check
    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    dec = get_encoder(enc_id)
    
    # generate verification metrics
    tmp = vf.parent.with_suffix('.tmp')
    tmp.mkdir(exist_ok=(not tmp_dir))

    r = dec.decode_variant(vd, a, tmp)
    if not a.dry_run:
        assert r.reconstructed_md5 == vd.reconstruction['md5']

    # compute tmp metrics & compare
    metrics_new = compute_metrics(a, vd, r)
    
    if a.dry_run:
        return

    report = {}
    for m in VariantMetricSet.get_keys():
        found = metrics_new[m]
        expected = vd.metrics[m]
        if math.isclose(found, expected, rel_tol=1e-4):
            report[m] = f'found:{found},expected:{expected}'
    
    vd.verification['date'] = datetime.now(timezone.utc).isoformat()
    vd.verification['verified'] = len(report) == 0
    vd.save_as(vf)
    
    # remove tmp files
    if tmp_dir:
        shutil.rmtree(tmp)

    return report

def main():
    
    anchors, references = parse_args()
    references_dir = Path('/media/akgrown/NILS/5GVIDEO/ReferenceSequences')
    # anchors_dir = Path('/media/akgrown/NILS/5GVIDEO/Anchors/Scenario-5')
    anchors_dir = Path('v2/5GVideo/Anchors/Scenario-5')

    refs = reference_sequences_by_keys(references, references_dir)

    for a in iter_anchors(anchors, refs, anchors_dir):
        assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
        a.dry_run = True
        for vf, vd in iter_variants(a):
            if vd == None:
                print('Not found:', vf)
                continue
            verify_metrics(vf, vd, a, tmp_dir=False)
            """
            try:
                _ = v.locate_bitstream(anchor_dir=a.working_dir, md5_check=True)
            except AssertionError as e:
                print("error", e)
                print("> encode, decode, metrics, save")
            
            if v.has_metric_set():
                print("> decode, metrics, compare")
            
            else:
                print("> decode, metrics, save")
            """

if __name__ == "__main__":
    main()
