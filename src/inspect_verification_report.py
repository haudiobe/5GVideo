import os
import sys
import ast
import click
import csv
from statistics import mean
from pathlib import Path
from sequences import TransferFunction
from anchor import AnchorTuple
from constants import BITSTREAMS_DIR, SEQUENCES_DIR
VCC_WORKING_DIR = Path(os.getenv('VCC_WORKING_DIR', '/data'))


def find_errors_in_report(csv_report:Path, threshold=0.01):

    def parse_info(info):
        l = ast.literal_eval(info)
        return {k: v for (k, v) in l}
    
    try:
        with open(csv_report) as fo:
            
            csv_reader = csv.DictReader(fo)
            errors = {}

            def add_error(vkey, ikey, ival, ival_orig):
                if vkey not in errors:
                    errors[vkey] = {}
                errors[vkey][ikey] = (ival, ival_orig)
                return errors
            
            for row in csv_reader:
                vkey = row['key']
                if row['status'] != 'successfull':
                    found = parse_info(row['info'])
                    orig = parse_info(row['orig-info'])
                    for k, v in found.items():
                        if k == "md5":
                            if v != orig[k]:
                                add_error(vkey, k, v, orig[k])
                        elif abs(v - orig[k]) > threshold:
                            add_error(vkey, k, v, orig[k])
            
            return errors

    except FileNotFoundError as e:
        print(e)
        return []


@click.command()
@click.option('--threshold', default=0.01, help='threshold above which a difference is reported as error')
@click.option('--outcsv', default=None, help='csv output')
@click.argument('REPORT')
def main(threshold, outcsv, report):
    csv_out = None if outcsv is None else Path(outcsv)
    csv_report = Path(report)
    threshold = float(threshold)
    
    errors = find_errors_in_report(csv_report, threshold)

    if len(errors):
        
        anchor_errs = {}

        for variant_key, details in errors.items():
            print("\n###", variant_key)
            parts = variant_key.split("-")
            anchor_key = "-".join(parts[0:-1])
            qp = variant_key.split("-")[-1]
            anc = anchor_errs[anchor_key] if anchor_key in anchor_errs else {}

            for k, v in details.items():
                if k != "md5":
                    err = abs((v[0])-v[1])
                    if k in anc:
                        anc[k].append((qp, err))
                    else:
                        anc[k] = [(qp, err)]
                print(f'* {k} - found: {v[0]} - expected: {v[1]}')
        
            anchor_errs[anchor_key] = anc


        if csv_out is not None:
            bitstreams_dir = VCC_WORKING_DIR / BITSTREAMS_DIR
            sequences_dir = VCC_WORKING_DIR / SEQUENCES_DIR
            rows = []
            for a, errs in anchor_errs.items():
                for k, v in errs.items():
                    qpk = [qp for qp, _ in v]
                    e  = [e for _, e in v]
                    err_meam = sum(e) / len(qpk)
                    anc = AnchorTuple.load(a, bitstreams_dir, sequences_dir)
                    rows.append({
                        "scenario": a.split("-")[0], 
                        "anchor": a,
                        "qp": qpk,
                        "metric": k,
                        "mean_error": err_meam,
                        "encoder_cfg": anc.encoder_cfg_key,
                        "sequence": Path(anc.reference.path).name,
                        "width": anc.reference.width,
                        # "height": anc.reference.height,
                        "frame_rate": anc.reference.frame_rate,
                        "bitdepth": anc.reference.bit_depth,
                        # "frame_count": anc.reference.frame_count,
                        # "matrix_coefficients": anc.reference.matrix_coefficients,
                        # "colour_primaries": anc.reference.colour_primaries,
                        # "chroma_format": anc.reference.chroma_format,
                        # "chroma_sample_loc_type": anc.reference.chroma_sample_loc_type,
                        # "chroma_subsampling": anc.reference.chroma_subsampling,
                        "transfer_characteristics": anc.reference.transfer_characteristics,
                        # "hdr_master_display": anc.reference.hdr_master_display,
                        # "hdr_max_fall": anc.reference.hdr_max_fall,
                        # "hdr_max_cll": anc.reference.hdr_max_cll,
                        # "interleaved": anc.reference.interleaved,
                        # "interlaced": anc.reference.interlaced
                    })
            exists = csv_out.exists()
            if len(rows) == 0:
                print("no errors found (md5 unknown)\n")
            else:
                with open(csv_out, newline='', mode='a+') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                    if not exists:
                        writer.writeheader()
                    cur_scenario=None
                    cur_anchor=None
                    for r in sorted(rows, key=lambda r: r["anchor"]):
                        if cur_scenario == r['scenario']:
                            r['scenario'] = '-'
                        else:
                            cur_scenario = r['scenario']
                        if cur_anchor == r['anchor']:
                            dup = ['anchor', 'qp', 'encoder_cfg','sequence','width','frame_rate','bitdepth','transfer_characteristics']
                            for k in dup:
                                r[k] = ''
                        else:
                            cur_anchor = r['anchor']
                        writer.writerow(r)


if __name__ == '__main__':
   main()
