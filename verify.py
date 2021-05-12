#!/usr/bin/python3
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone 
from enum import Enum
import math

from typing import Any, List, Iterable, Tuple

from anchor import AnchorTuple, VariantData, reference_sequences_dict, iter_anchors, iter_variants
from encoders import get_encoder
from metrics import compute_metrics, VariantMetricSet


class AnchorVerification(Enum):
    BITSTREAM = 1
    DECODER = 2


def save_verification_report(vf:Path, vd:VariantData, verification_type:AnchorVerification, success:bool, log_data:List[str]=None, contact:str=None):

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
            fo.writelines(log_data)
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


def verify_variant_metrics(a:AnchorTuple, vd:VariantData, vf:Path, tmp_dir:Path=None, debug=True) -> Tuple[bool, str]:
    """
    1. decode into a temporary folder
    2. compute metrics with freshly decoded data
    3. compare with values defined in VariantData
    return a diff in case of missmatch
    """
    
    log = [f'= {vd.variant_id} | decoder verification =====\n']
    success = True

    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    dec = get_encoder(enc_id)

    if tmp_dir == None:
        tmp = vf.parent.with_suffix('.tmp')
    else:
        tmp = tmp_dir / vf.stem

    tmp.mkdir(exist_ok=(not debug))

    compute_md5 = vd.reconstruction['md5'] != 'unknown'
    r = dec.decode_variant(a, vd, tmp, md5=compute_md5)
    
    metrics_new = compute_metrics(a, vd, r).to_dict()

    """
    if a.dry_run:
        return False, "/!\\ dry run"
    """

    md5_new = r.reconstructed_md5
    md5_ref = vd.reconstruction['md5']
    if md5_new != md5_ref:
        log.append(f'md5 mismatch - expected:{md5_ref} - result: {md5_new}\n')
        success = False
    
    for m in VariantMetricSet.get_keys(a):
        found = metrics_new[m]
        expected = vd.metrics[m]
        if math.isclose(found, expected, rel_tol=1e-4):
            success = False
        log.append(f'{m} - expected:{expected} - result: {found}\n')
    
    if not debug:
        shutil.rmtree(tmp)
    
    return success, log


def verify_anchor_metrics(a:AnchorTuple, contact:str=None, tmp_dir:Path=None):
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        success, log_data = verify_variant_metrics(a, vd, vf, tmp_dir=tmp_dir)
        save_verification_report(vf, vd, AnchorVerification.DECODER, success, log_data, contact=contact)
        if not success:
            a_errors.append(log_data)
    return a_errors


#########################################################################################################

def bitstream_verification_preflight(a:AnchorTuple) -> List[Tuple[Path, Any, str]]:
    """
    verifies:
        * all variant bitstreams exists
        * all bitstreams md5 match their json description
    """
    err = None
    for vf, vd in iter_variants(a):
        try:
            _ = vd.locate_bitstream(a.working_dir, md5_check=True)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            if err == None:
                err = []
            err.append((vf, vd, e))
    cfg = a.encoder_cfg
    if not cfg.exists():
        if err == None:
               err = []
        err.append((cfg, a, 'encoder configuration not found'))
    return err


def verify_variant_bitstream(a:AnchorTuple, vd:VariantData, vf:Path, tmp_dir:Path=None, debug=True) -> Tuple[bool, str]:
    """
    verifies:
        * re-encoding a new bitstream matches md5 of the existing bitstream
    """
    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    enc = get_encoder(enc_id)

    if tmp_dir == None:
        tmp = vf.parent.with_suffix('.tmp')
    else:
        tmp = tmp_dir / vf.stem

    tmp.mkdir(exist_ok=(not debug))

    vd_new = enc.encode_variant(a, vd.variant_id, vd.variant_cli, tmp)
    md5_new = vd_new.bitstream["md5"]
    md5_ref = vd.bitstream["md5"]
    
    if not debug:
        shutil.rmtree(tmp)

    if md5_new == md5_ref:
        return True
    else:
        return False, f'invalid md5 - expected:{md5_new} - found:{md5_ref}'


def verify_anchor_bitstreams(a:AnchorTuple, contact:str=None, tmp_dir:Path=None):
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        success, log_data = verify_variant_bitstream(a, vd, vf, tmp_dir=tmp_dir)
        save_verification_report(vf, vd, AnchorVerification.BITSTREAM, success, log_data, contact=contact)
        if not success:
            a_errors.append(log_data)
    return a_errors


#########################################################################################################

def verify(verification_type:AnchorVerification, anchors:Iterable[AnchorTuple], dry_run=True, contact=None, tmp_dir:Path=None):

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
        errors = verification_fn(a, contact, tmp_dir=tmp_dir)
        print(f'{a} - error - verification failed:\n * {errors}')



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='"bitstream" or "decoder"')
    parser.add_argument('-s','--sequences', required=True, type=str, help='sequences.csv file containing the list of reference raw sequences')
    parser.add_argument('-a', '--anchors', required=True, type=str, help='anchors.csv file containing the list of anchors for a scenario')
    parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
    parser.add_argument('--s_dir', required=False, type=str, help='the directory containing the reference sequences')
    parser.add_argument('--a_dir', required=False, type=str, help='a scenario directory, containing the anchors')
    parser.add_argument('--cfg_dir', required=False, type=str, help='directory where encoder configurations can be found, required for bitstream verification')
    parser.add_argument('-c', '--contact', required=True, type=str, help='email for the verification report contact field')
    parser.add_argument('--tmp_dir', required=False, type=str, help='tmp directory to encode / decode data')
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()
    
    references_csv = Path(args.sequences)
    assert references_csv.exists()

    if args.s_dir != None:
        references_dir = Path(args.s_dir)
        assert references_dir.is_dir()
    else:
        references_dir = references_csv.parent

    anchors_csv = Path(args.anchors)
    if args.a_dir != None:
        scenario_dir = Path(args.a_dir)
        assert scenario_dir.is_dir()
    else:
        scenario_dir = anchors_csv.parent

    if args.type == "bitstream":
        verification_type = AnchorVerification.BITSTREAM
        err = (args.cfg_dir == None)
        if not err:
            err = not Path(args.cfg_dir).exists()
        if err:
            parser.error("`--cfg_dir` is required for bitstream verification")
    elif args.type == "decoder":
        verification_type = AnchorVerification.DECODER
    else:
        parser.error("invalid verification type")
        return

    tmp_dir = None
    if args.tmp_dir != None:
        tmp_dir = Path(args.tmp_dir)
        assert tmp_dir.exists()

    keys = None
    if args.key != None:
        keys = [args.key]

    refs = reference_sequences_dict(references_csv, references_dir)

    anchors = iter_anchors(anchors_csv, refs, scenario_dir, Path(args.cfg_dir), keys=keys)
    try:
        verify(
            verification_type,
            anchors, 
            dry_run=args.dry_run,
            contact=args.contact,
            tmp_dir=tmp_dir
        )
    except KeyboardInterrupt as e:
        print(e)
        return

if __name__ == "__main__":
    main()
