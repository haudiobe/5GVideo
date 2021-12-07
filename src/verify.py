import logging

import json
import click
from pathlib import Path
from datetime import datetime, timezone 
from enum import Enum
import math
from csv import DictWriter
from typing import List, Iterable
from anchor import VariantData, iter_anchors, load_variants
from metrics import HDR_METRICS, SDR_METRICS

def date_str():
    return datetime.now(timezone.utc).strftime('%d/%m/%y')

class AnchorVerificationType(Enum):
    BITSTREAM = 1
    DECODER = 2

class AnchorVerificationReport:

    CSV_FIELDNAMES = ["key", "file", "origdate", "md5", "status", "company", "e-mail", "vdate", "document", "type", "information"]

    def __init__(self, variant_data:VariantData, verification_type:AnchorVerificationType, success:bool, log=List[str], template:dict = None, log_file:Path = None) -> None:

        self.vd = variant_data
        if template == None:
            self.report = {
                "Contact": {
                    "Company": "",
                    "name": "",
                    "e-mail": ""
                },
                "date": date_str(),
                "meeting": "",
                "input": "",
                "information": ""
            }
        else:
            self.report = template
        
        if log_file is not None:
            self.report["information"] = str(log_file)
        
        if success:
            self.report["status"] = "failed"
        else:
            self.report["status"] = "successful"

        if verification_type == AnchorVerificationType.BITSTREAM:
            self.report["type"] = "bistream"
        elif verification_type == AnchorVerificationType.DECODER:
            self.report["type"] = "decoder"

        self.log = log
        self.log_file = log_file


    def save_error_log(self):
        if len(self.log):
            return
        assert self.log_file is not None, "log_file attribute not specified"
        with open(self.log_file, 'w') as fo:
            fo.writelines(self.log)


    @classmethod
    def get_csv_writer(cls, fo, writeheader=True) -> DictWriter:
        dw = DictWriter(fo, fieldnames=cls.CSV_FIELDNAMES)
        if writeheader:
            dw.writeheader()
        return dw


    def to_csv_dict(self):
        return {
            "key": self.vd.variant_id,
            "file": self.vd.bitstream['URI'],
            "origdate": self.vd.bitstream["date"],
            "md5": self.vd.bitstream['md5'],
            "status": self.report["status"],
            "company": self.report["Contact"]["Company"],
            "e-mail": self.report["Contact"]["e-mail"],
            "vdate": self.report["date"],
            "document": self.report["input"],
            "type": self.report["type"],
            "information": self.report["information"]
        }


def save_streams_verification(rl: List[AnchorVerificationReport], dst:Path) -> Iterable[Iterable[str]]:
    with open(dst) as fo:
        csvw = AnchorVerificationReport.get_csv_writer(fo)
        csvw.writerows(rl)
        for r in rl:
            r.to_csv_dict()
            csvw.writerows()


@click.group()
@click.pass_context
@click.option('--template', required=False, default = None, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('downloaded', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('local', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def cli(ctx, template:Path, downloaded:str, local:str):
    ctx.ensure_object(dict)

    downloaded = Path(downloaded)
    local = Path(local)

    if template is not None:
        with open(template, 'r') as fo:
            ctx.obj['template'] = json.load(fo)
    else:
        ctx.obj['template'] = None
    ctx.obj['anchor_streams'] = iter_anchors(downloaded, sequences=None)
    ctx.obj['test_streams'] = iter_anchors(local, sequences=None)



@cli.command()
@click.pass_context
def bitstream(ctx):
    """
    verify that bistream md5 are matching
    """
    reports = []
    for a, t in zip(ctx.obj['anchor_streams'], ctx.obj['test_streams']):
        av = load_variants(a)
        tv = load_variants(t)
        for (avf, avd), (_, tvd) in zip(av, tv):
            print(avf, avd)
            print(_, tvd)
            assert avd.variant_id == tvd.variant_id, "variant id missmatch, the 2 streams list are not equal"
            avd_md5 = avd.bitstream['md5']
            tvd_md5 = tvd.bitstream['md5']
            success = avd_md5 == tvd_md5
            log_msg = f'bitstream md5 - expected:{avd_md5} - result: {tvd_md5}\n)'
            logf = avf.parent / f'{avd.variant_id}_{date_str()}_decoder_verification.log'
            r = AnchorVerificationReport(avd, AnchorVerificationType.BITSTREAM, success, [log_msg], ctx.obj['template'], logf)
            reports.append(r)
    save_streams_verification(reports, a.working_dir / 'verification.csv')


@cli.command()
@click.pass_context
def decoder(ctx):
    """
    verify that reconstruction md5 and metrics are matching
    """
    reports = []

    for a, t in zip(ctx.obj['anchor_streams'], ctx.obj['test_streams']):
        av = load_variants(a)
        tv = load_variants(t)
        metrics = a.get_metrics_set()
        for (avf, avd), (_, tvd) in zip(av, tv):

            assert avd.variant_id == tvd.variant_id, "variant id missmatch, the 2 streams list are not equal"
            log = []
            success = True
            
            if (avd.reconstruction['md5'] == 'unknown') or (avd.reconstruction['md5'] == None):
                logging.warning('reconstruction md5 not specified')
            else:
                avd_md5 = avd.reconstruction['md5']
                tvd_md5 = tvd.reconstruction['md5']
                if avd_md5 != tvd_md5:
                    success = False
                log.append(f'reconstruction md5 - expected:{avd_md5} - result: {tvd_md5}\n)')

            for key, expected in avd.metrics.items():
                if key in metrics:
                    found = tvd.metrics[key]
                    if not math.isclose(found, expected, rel_tol=1e-4):
                        success = False
                    log.append(f'{key} - expected:{expected} - result: {found}\n)')
                logf = avf.parent / f'{avd.variant_id}_{date_str()}_decoder_verification.log'
                r = AnchorVerificationReport(avd, AnchorVerificationType.DECODER, success, log, ctx.obj['template'], logf)
                reports.append(r)
            
            save_streams_verification(reports, a.working_dir / 'verification.csv')



if __name__ == "__main__":
    cli()
