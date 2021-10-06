#!.venv/bin/python3

import argparse
from os import close, error, replace
import shutil
from pathlib import Path
from datetime import datetime, timezone 
from enum import Enum
import math
import json
from csv import DictWriter
import copy

from typing import Any, List, Iterable, Tuple

from anchor import AnchorTuple, VariantData, reference_sequences_dict, iter_anchors, iter_variants
from encoders import get_encoder, parse_encoding_bitdepth
from metrics import SDR_METRICS, Metric, compute_metrics, anchor_metrics_to_csv
from convert import as_10bit_sequence

DEBUG_FIRST_VARIANT_ONLY=False 
DEBUG_SKIP_VMAF=False

class AnchorVerificationCmd(Enum):
    REPORT = -1
    BITSTREAM = 1
    DECODER = 2

class AnchorVerificationReport:
    
    CSV_FIELDNAMES = [ "key", "file", "origdate", "md5", "status", "company", "e-mail", "vdate", "document", "type", "information" ]

    @classmethod
    def from_json_dict(cls, report):
        
        r = cls()

        contact = report["Contact"]
        r.contact_company = contact.get("Company", None)
        r.contact_e_mail = contact.get("e-mail", None)

        r.meeting = report.get("meeting", None)
        r.input_doc = report.get("input", None)
        r.status = report.get("status", None)
        r.type = report.get("type", None)
        r.information = report.get("information", None)

        r.verification_date = report.get("date", None)

        return r

    @classmethod
    def get_csv_writer(cls, fo, writeheader=True) -> DictWriter:
        dw = DictWriter(fo, fieldnames=cls.CSV_FIELDNAMES)
        if writeheader:
            dw.writeheader()
        return dw

    def to_csv_dict(self, vf:Path, vd:VariantData, template:"AnchorVerificationReport"=None):
        row = {
            "key":vd.variant_id,
            "file":str(vf),
            "origdate":vd.bitstream["date"],
            "md5":vd.bitstream["md5"],
            "status":self.status,
            "company":self.contact_company,
            "e-mail":self.contact_e_mail,
            "vdate":self.verification_date,
            "document":self.input_doc,
            "type":self.type,
            "information":self.information
        }
        if template != None:
            row["company"] = template.contact_company
            row["e-mail"] = template.contact_e_mail
            row["document"] = template.input_doc
        return row



def save_verification_report(vf:Path, vd:VariantData, verification_type:AnchorVerificationCmd, success:bool, log_data:List[str]=None, template:dict=None):
    
    report = template
    ts = datetime.now(timezone.utc)

    if report == None:
        report = {
            "Contact": {
                "Company": "",
                "name": "",
                "e-mail": ""
            },
            "date": ts.strftime('%d/%m/%y'),
            "meeting": "",
            "input": ""
        }

    if not success:
        assert (log_data != None) and (log_data != '')
        report["status"] = "failed"
    else:
        report["status"] = "successful"

    if vd.verification == None:
        vd.verification = { "Reports": [] }
    elif not "Reports" in vd.verification:
        vd.verification["Reports"] = []
    
    idx = len(vd.verification["Reports"])

    if verification_type == AnchorVerificationCmd.BITSTREAM:
        report["type"] = "bistream"
    elif verification_type == AnchorVerificationCmd.DECODER:
        report["type"] = "decoder"

    if log_data:
        # save verification report
        log_file = vf.parent / f'{vd.variant_id}_verification_{idx}_{report["type"]}_{ts.strftime("%d%m%y")}.log'
        with open(log_file, 'w') as fo:
            fo.writelines(log_data)
        report["information"] = str(Path('.') / log_file.name)
    else:
        report["information"] = ''

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
            _ = a.locate_bitstream(vd, md5_check=True)
            vd.has_metric_set( SDR_METRICS )
        except BaseException as e:
            if err == None:
                err = []
            err.append((vf, vd, e))
    return err


def verify_variant_metrics(a:AnchorTuple, vd:VariantData, vf:Path, tmp_dir:Path=None, debug=True, reconstrution_md5=False) -> Tuple[bool, str]:
    """
    1. decode into a temporary folder
    2. compute metrics with freshly decoded data
    3. compare with values defined in VariantData
    return a diff in case of missmatch
    """
    coded_bit_depth = parse_encoding_bitdepth(a.encoder_cfg)
    if (a.reference.bit_depth == 8) and (coded_bit_depth == 10):
        assert as_10bit_sequence(a.reference).path.exists()
    assert vd.metrics != None, f'no metrics defined for variant {a.anchor_key}/{vd.variant_id}.json'

    log = [f'= {vd.variant_id} | decoder verification =====\n']
    success = True

    enc_id, enc, dec = a.encoder_id, vd.generation['encoder'], vd.reconstruction['decoder']
    assert enc_id == dec and enc == dec, f'encoder id mismatch: {(enc_id, enc, dec)}'
    dec = get_encoder(enc_id)

    if tmp_dir == None:
        tmp = vf.parent.with_suffix('.tmp')
    else:
        tmp = tmp_dir / vf.parent.with_suffix('.tmp').name
    
    if not a.dry_run:
        tmp.mkdir(exist_ok=True)

    verify_reconstruction_md5 = reconstrution_md5 and (vd.reconstruction['md5'] != 'unknown')
    r = dec.decode_variant(a, vd, tmp, md5=reconstrution_md5)
    print("decoded")
    _, metrics_new = compute_metrics(a, vd, vmaf=(not DEBUG_SKIP_VMAF), dist_dir=tmp)

    if a.dry_run:
        return False, "/!\\ dry run"

    if verify_reconstruction_md5:
        md5_new = r.reconstructed_md5
        md5_ref = vd.reconstruction['md5']
        if md5_new != md5_ref:
            log.append(f'md5 mismatch - expected:{md5_ref} - result: {md5_new}\n')
            success = False
        
    skipped = [Metric.DECODETIME.key, Metric.ENCODETIME.key]
    for key, expected in vd.metrics.items():
        if key in skipped:
            continue
        found = metrics_new[key]
        if math.isclose(found, expected, rel_tol=1e-4):
            success = False
        log.append(f'{key} - expected:{expected} - result: {found}\n')
    
    if not debug:
        shutil.rmtree(tmp)
    
    return success, log


def verify_anchor_metrics(a:AnchorTuple, template:dict=None, tmp_dir:Path=None) -> Iterable[Iterable[str]]:
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        print('-'*128)
        assert vd != None, f'variant data not found:{vf}'
        success, log_data = verify_variant_metrics(a, vd, vf, tmp_dir=tmp_dir)
        if not a.dry_run:
            save_verification_report(vf, vd, AnchorVerificationCmd.DECODER, success, log_data, template=template)
        if not success:
            a_errors.append(log_data)
        if DEBUG_FIRST_VARIANT_ONLY:
            break
    if len(a_errors) == 0:
        anchor_metrics_to_csv(a)
    print('-'*128)
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
            _ = a.locate_bitstream(vd, md5_check=True)
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

    if not a.dry_run:
        tmp.mkdir(exist_ok=True)

    vd_new = enc.encode_variant(a, vd.variant_id, vd.variant_cli, tmp)

    if a.dry_run:
        return True, None

    md5_new = vd_new.bitstream["md5"]
    md5_ref = vd.bitstream["md5"]
    
    if not debug:
        shutil.rmtree(tmp)

    if md5_new == md5_ref:
        return True, None
    else:
        return False, f'invalid md5 - expected:{md5_new} - found:{md5_ref}'


def verify_anchor_bitstreams(a:AnchorTuple, template:dict=None, tmp_dir:Path=None) -> Iterable[Iterable[str]]:
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    a_errors = []
    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        print('-'*128)
        success, log_data = verify_variant_bitstream(a, vd, vf, tmp_dir=tmp_dir)
        if not a.dry_run:
            save_verification_report(vf, vd, AnchorVerificationCmd.BITSTREAM, success, log_data, template=template)
        if not success:
            a_errors.append(log_data)
        if DEBUG_FIRST_VARIANT_ONLY:
            break
    print('-'*128)
    return a_errors


def report_anchor_verifications(a:AnchorTuple, dw:DictWriter, template:dict=None) -> Iterable[Iterable[str]]:
    assert a.working_dir.is_dir(), f'invalid anchor directory: {a.working_dir}'
    if template != None:
        template = AnchorVerificationReport.from_json_dict(template)

    for vf, vd in iter_variants(a):
        assert vd != None, f'variant data not found:{vf}'
        data = vd.verification["Reports"][-1]
        report = AnchorVerificationReport.from_json_dict(data)
        row = report.to_csv_dict(vf, vd, template=template)
        dw.writerow(row)



#########################################################################################################

def verify(verification_type:AnchorVerificationCmd, anchors:Iterable[AnchorTuple], dry_run=True, template:dict=None, tmp_dir:Path=None, report_dir:Path=None):

    batch = []
    preflight_errors = {}

    if verification_type == AnchorVerificationCmd.DECODER:
        preflight_fn = decoder_verification_preflight
        verification_fn = verify_anchor_metrics

    elif verification_type == AnchorVerificationCmd.BITSTREAM:
        preflight_fn = bitstream_verification_preflight
        verification_fn = verify_anchor_bitstreams

    for a in anchors:
        err = preflight_fn(a)
        if err == None:
            batch.append(a)
        else:
            preflight_errors[a] = err
            print(f'# {a.anchor_key} - error - can not run verification:\n#{err}\n')
            batch.append(a)

    for a in batch:
        a.dry_run = dry_run
        errors = verification_fn(a, template, tmp_dir=tmp_dir)
        print(f'# {a.anchor_key} verification complete')
        print(f'# {len(errors)} errors')
        if len(errors) and not dry_run:
            for err in errors:
                print(f'#\t* {err}')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='"bitstream" or "decoder"')
    parser.add_argument('--scenario_dir', required=True, type=str, help='scenario directory')
    parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
    parser.add_argument('-a', '--anchors-list', required=False, type=str, default='./streams.csv', help='streams.csv file containing the list of anchors for a scenario')
    parser.add_argument('-s','--sequences-list', required=False, type=str, default='../reference-sequence.csv', help='sequences.csv file containing the list of reference raw sequences')
    parser.add_argument('--sequences_dir', required=False, type=str, help='the directory containing the reference sequences')
    parser.add_argument('--cfg_dir', required=False, type=str, help='directory where encoder configurations can be found, required for bitstream verification')
    parser.add_argument('--template', required=False, type=str, help='json template for the verification report')
    parser.add_argument('--tmp_dir', required=False, type=str, help='tmp directory to encode / decode data')
    parser.add_argument('--dry-run', action='store_true', default=False)
    
    args = parser.parse_args()
    
    scenario_dir = Path(args.scenario_dir)
    assert scenario_dir.is_dir(), f'invalid scenario directory {scenario_dir}'
    
    anchors_csv = scenario_dir / args.anchors_list
    assert anchors_csv.exists(), f'anchor list not found {anchors_csv}'
    
    references_csv = scenario_dir / Path(args.sequences_list)
    assert references_csv.exists(), f'reference list not found {references_csv}'

    if args.sequences_dir:
        assert Path(args.sequences_dir).is_dir(), f'invalid sequence directory {args.sequences_dir}'
        sequences_dir = Path(args.sequences_dir)
    else:
        try:
            sequences_dir = scenario_dir.parent.parent.parent / 'ReferenceSequences'
        except:
            assert 0, f'invalid sequence directory. check the --sequences_dir option'
    
    cfg_dir = scenario_dir / '../CFG'
    if args.type == "bitstream":
        verification_type = AnchorVerificationCmd.BITSTREAM
        if args.cfg_dir != None:
            cfg_dir = args.cfg_dir
            if not cfg_dir.exists():
                parser.error("`a valid --cfg_dir` is required for bitstream verification")

    elif args.type == "decoder":
        verification_type = AnchorVerificationCmd.DECODER

    elif args.type == "report":
        verification_type = AnchorVerificationCmd.REPORT

    else:
        parser.error("invalid verification type")
        return

    tmp_dir = None
    if args.tmp_dir != None:
        tmp_dir = Path(args.tmp_dir)
        assert tmp_dir.exists(), 'invalid temporary dir'

    keys = None
    if args.key != None:
        keys = [args.key]

    refs = reference_sequences_dict(references_csv, sequences_dir)
    anchors = iter_anchors(anchors_csv, refs, scenario_dir, keys=keys)

    print('# PROCESSING', len(anchors), 'anchors', '#'*32)
    if len(anchors) == 0:
        return

    # verification report template
    template = None
    if args.template:
        with open(args.template, 'r') as fo:
            template = json.load(fo)
    
    if verification_type == AnchorVerificationCmd.REPORT:
        fp = scenario_dir / 'verification_report.csv'
        with open(fp, 'w') as fo:
            print('saving to: ', fp)
            dw = AnchorVerificationReport.get_csv_writer(fo)
            for a in anchors:
                print('writing verifications for: ', a.anchor_key )
                report_anchor_verifications(a, dw, template)

    else:
        try:
            verify(
                verification_type,
                anchors, 
                dry_run=args.dry_run,
                template=template,
                tmp_dir=tmp_dir
            )
        except KeyboardInterrupt as e:
            print(e)
            return

if __name__ == "__main__":
    main()
