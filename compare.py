
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any
import matplotlib
from numpy.core.shape_base import block

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np

from anchor import VariantData, VariantMetricSet, Metric, iter_variants, AnchorTuple
from download import AnchorTupleCtx
from metrics import SDR_METRICS, BD_RATE, Metric, compute_avg_psnr, compute_log_msssim
import sys, csv

def rounded(v):
    return f'{round(v, 2):.2f}'

def rd_plot(r0, d0, r1, d1, dist='psnr', anchor_label='anchor', test_label='test', title='', show=True):
    if title == '':
        title = f'{dist} rd curve'
    fig, axs = plt.subplots(1, 1, figsize=(20,15))
    axs.plot(r0, d0, 'o-', r1, d1, 'o-')
    axs.set_xlabel('bitrate', fontsize=21)
    axs.set_ylabel(dist, fontsize=21)
    axs.grid(True)
    axs.tick_params(axis='both', which='major', labelsize=21)
    axs.set_title(title, fontdict={'fontsize': 24, 'fontweight': 'medium'})
    axs.legend([anchor_label, test_label])
    if show:
        plt.show(block=True)
    return fig

def rd_metrics(variants:List[VariantData], rate="BitrateLog", dist="PSNR") -> Iterable[Any]:
        return zip(*[(v.metrics[rate], v.metrics[dist]) for v in variants])

def compare_anchors_metrics( anchor:List[VariantData], test:List[VariantData], rate="BitrateLog", dist="PSNR", piecewise=1, strict=False, sanitize=True ) -> float:
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    try:
        print("#", dist, "#"*(128-len(str(dist))))
        return BD_RATE(*anchor_metrics, *test_metrics, piecewise=piecewise, sanitize=sanitize)
    except BaseException as e:
        if strict:
            raise
        raise ValueError(f'A:{anchor_metrics} | T:{test_metrics} | Err: {e}')

def variant_metrics_to_csv(variants, path='./variants.csv', csv_append=False, csv_headers=True):
    with open(path, 'a' if csv_append else 'w', newline='') as csvfile:
        fieldnames = [
            "ID",
            "BitrateLog",
            "YPSNR",
            "UPSNR",
            "VPSNR",
            "MS_SSIM",
            "VMAF",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_headers:
            writer.writeheader()
        for v in variants:
            row = { k: v for [k, v] in v.metrics.items() if k in fieldnames }
            row['ID'] = v.variant_id
            writer.writerow( row )


def compare_anchors_directories(refdir:Path, testdir:Path, metrics:Iterable[str], strict=False, sanitize=True, plots=None) -> Iterable[Tuple[str, VariantMetricSet]]:

    def load_anchor_directory(ad:Path) -> Tuple[AnchorTuple, List[VariantData]]:
        assert ad.is_dir(), f'not a directory {ad}'
        ctx, anchor_key = AnchorTupleCtx.from_anchor_directory(ad)
        a = ctx.iter_anchors(keys=[anchor_key])[0]
        v = [ vdata for (_, vdata) in iter_variants(a) ]
        return a, v

    aref, vref = load_anchor_directory(refdir)
    atest, vtest = load_anchor_directory(testdir)

    vms = VariantMetricSet()
    for key in metrics:
        try:
            bd, r0, d0, r1, d1 = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key, strict=strict, sanitize=sanitize)
            vms[key] = f'{round(bd, 3):.3f}'
            if plots and (key in plots):
                t = f'bd-rate @{key} | {vms[key]}'
                if sanitize:
                    t += ' [sanitized]'
                fig = rd_plot(r0, d0, r1, d1, key, title=t, anchor_label=aref.anchor_key, test_label=atest.anchor_key, show=False)
                fname = testdir.parent / 'Metrics' / f'{refdir.name}.{testdir.name}.{key}.png'.lower()
                fig.savefig(fname)
        except BaseException as e:
            if strict:
                raise
            vms[key] = str(e)
    
    return aref.reference.sequence['Key'], vms


#######################################################

def _parse_filter(fp:Path):
    arr = fp.name.split("@")
    if len(arr) > 1:
        return fp.parent / "@".join(arr[:-1]), arr[-1]
    else:
        return fp, None

def compare_encoder_configs(ref:Path, test:Path, metrics:List[str], strict=False):

    refdir, refkey = _parse_filter(ref)
    refctx = AnchorTupleCtx(scenario_dir=refdir)
    ref_anchors = refctx.iter_anchors(cfg_keys=[refkey])
    
    testdir, testkey = _parse_filter(test)
    testctx = AnchorTupleCtx(scenario_dir=testdir)
    test_anchors = testctx.iter_anchors(cfg_keys=[testkey])

    assert len(ref_anchors) == len(test_anchors), f'reference has {len(ref_anchors)} anchors, test has {len(test_anchors)} anchors'

    bd_rates = []

    for aref, atest in zip(ref_anchors, test_anchors):
        vref = [ v for (_, v) in iter_variants(aref) ]
        vtest = [ v for (_, v) in iter_variants(atest) ]
        arates = {}
        for key in metrics:
            try:
                bd, *_  = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
                arates[key] = bd
            except BaseException as e:
                if strict:
                    raise
                arates[key] = str(e)

        bd_rates.append((aref, atest, arates))

    return [(r.reference.sequence['Key'], bdr) for (r, _, bdr) in bd_rates]

def csv_dump(data, fp):
    fieldnames = [ 
        'reference',
        Metric.PSNR_Y.key,
        Metric.PSNR_U.key,
        Metric.PSNR_V.key,
        Metric.PSNR.key,
        Metric.MSSSIM.key,
        Metric.VMAF.key
    ]

    stats = {
        'min': {},
        'max': {},
        'avg': {},
    }

    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    with open(fp, 'w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in data:
            
            row = {}

            for k in fieldnames:
                row[k] = r[k]
                if len(data) == 1:
                    continue
                if k == "reference":
                    continue
                
                v = row[k]
                if v == None or type(v) == str:
                    continue
                if k in stats["min"]:
                    stats["min"][k] = min(v, stats["min"][k])
                else:
                    stats["min"][k] = v
                if k in stats["max"]:
                    stats["max"][k] = max(v, stats["max"][k])
                else:
                    stats["max"][k] = v
                if k in stats["avg"]:
                    stats["avg"][k].append(v)
                else:
                    stats["avg"][k] = [v]
            
            writer.writerow(row)

        if len(data) == 1:
            return

        r = { "reference": "Min" }
        for k in fieldnames[1:]:
            r[k] = stats["min"].get(k, None)
        writer.writerow(r)

        r = { "reference": "Max" }
        for k in fieldnames[1:]:
            r[k] = stats["max"].get(k, None)
        writer.writerow(r)

        r = { "reference": "Avg" }
        for k in fieldnames[1:]:
            avg = None
            if k in stats["avg"] and len(stats["avg"][k]):
                avg = 0
                for j in stats["avg"][k]:
                    avg += j
                avg /= len(stats["avg"][k])
            r[k] = avg
        writer.writerow(r)


def main():

    assert len(sys.argv) == 3

    ref = Path(sys.argv[1])
    test = Path(sys.argv[2])

    print("Anchor:", ref)
    print("Test:", test)
    
    metrics = [str(m.key) for m in SDR_METRICS]

    if ref.is_dir() and (ref.parent / 'streams.csv').exists() \
        and test.is_dir() and (test.parent / 'streams.csv').exists():
        # eg. compare.py ./scenario/codec1/a_key1 ./scenario/codec2/a_key2
        plots = [ m.key for m in (
            Metric.PSNR_Y, 
            Metric.PSNR_U, 
            Metric.PSNR_V, 
            Metric.PSNR, 
            Metric.MSSSIM, 
        )]
        seqid, r = compare_anchors_directories(ref, test, metrics, strict=True, sanitize=False, plots=plots)
        r['reference'] = seqid
        outp = test.parent / 'Metrics' / f'{ref.name}.{test.name}.csv'.lower()
        csv_dump([r], outp)
    else:
        # eg. compare.py ./scenario/codec1@cfg_id1 ./scenario/codec2@cfg_id2
        _, refkey = _parse_filter(ref)
        testdir, testkey = _parse_filter(test)
        outp = testdir / 'Metrics' / f'{refkey}.{testkey}.csv'.lower()
        data = []
        for (seqid, r) in compare_encoder_configs(ref, test, metrics, strict=True):
            r['reference'] = seqid
            data.append(r)
        csv_dump(data, outp)
        
if __name__ == "__main__":
    main()