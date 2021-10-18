
from pathlib import Path
from typing import Iterable, List, Tuple, Any
import matplotlib.pyplot as plt

# plt.ioff()

import numpy as np
import scipy.interpolate

from anchor import VariantData, VariantMetricSet, Metric, iter_variants, AnchorTuple
from download import AnchorTupleCtx
from metrics import SDR_METRICS, BD_RATE, Metric, sanitize_rd_data1, sanitize_rd_data2, sort_on_rates
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

def compare_anchors_metrics( anchor:List[VariantData], test:List[VariantData], rate="BitrateLog", dist="psnr", title="", strict=False, sanitize=True ) -> float:
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    try:
        print("#", dist, "#"*(32-len(str(dist))))
        # return BD_RATE(*anchor_metrics, *test_metrics, piecewise=piecewise, sanitize=sanitize)
        return BD_RATE_PLOT(*anchor_metrics, *test_metrics, sanitize=sanitize, title=title, dist_label=dist)
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
    fig = None
    for key in metrics:
        try:
            plot_title = f'RD-curve & BD-rate for {key}\n'\
                f'[sequence] {aref.reference.sequence["Key"]} | {aref.reference.sequence["Name"]}\n'\
                f'[anchor] {aref.encoder_cfg.name} @ {aref._variants}\n'\
                f'[test] {atest.encoder_cfg.name} @ {atest._variants}'
            fig, bd, *_ = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key, strict=strict, sanitize=sanitize, title=plot_title)
            vms[key] = f'{round(bd, 3):.3f}'
            if plots and (key in plots):
                # t = f'bd-rate @{key} | {vms[key]}'
                # if sanitize:
                #     t += ' [sanitized]'
                # fig = rd_plot(r0, d0, r1, d1, key, title=t, anchor_label=aref.anchor_key, test_label=atest.anchor_key, show=False)
                fname = testdir.parent / 'Metrics' / f'{refdir.name}.{testdir.name}.{key}.png'.lower()
                fig.savefig(fname)
                plt.close()

        except BaseException as e:
            if fig:
                fig.close()
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

    assert len(ref_anchors) == len(test_anchors), f'reference has {len(ref_anchors)} anchors, test has {len(test_anchors)} anchors.'

    bd_rates = []

    for aref, atest in zip(ref_anchors, test_anchors):
        vref = [ v for (_, v) in iter_variants(aref) ]
        vtest = [ v for (_, v) in iter_variants(atest) ]
        arates = {}
        print("\n"+"="*35)
        print(f'{aref.anchor_key} vs {atest.anchor_key}\n'+'='*35)
        for key in metrics:
            try:
                _, bd, *_  = compare_anchors_metrics(vref, vtest, rate="BitrateLog", dist=key)
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



#####################################################################################################


def BD_RATE_PLOT(R1, PSNR1, R2, PSNR2, sanitize=False, title="", dist_label="dist"):

    if sanitize:
        R1, PSNR1 = sanitize_rd_data1(R1, PSNR1)
        R2, PSNR2 = sanitize_rd_data1(R2, PSNR2)
    else:
        PSNR1 = np.array(PSNR1)
        PSNR2 = np.array(PSNR2)

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
    [y1, x1] = sort_on_rates(lR1, PSNR1)
    [y2, x2] = sort_on_rates(lR2, PSNR2)
    err = None
    fig = None
    avg_diff = None
    try:
        v1 = scipy.interpolate.pchip_interpolate(x1, y1, samples)
        v2 = scipy.interpolate.pchip_interpolate(x2, y2, samples)

        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1) * -100

        # plot it
        fig, axs = plt.subplots(2, 1, figsize=(10, 20))

        axs[0].plot(R1, PSNR1, 'o-', R2, PSNR2, 'o-')
        axs[0].set_xlabel('bitrate', fontsize=21)
        axs[0].set_ylabel(dist_label, fontsize=21)
        axs[0].grid(True)
        axs[0].tick_params(axis='both', which='major', labelsize=21)
        axs[0].set_title('RD curve', fontdict={'fontsize': 24, 'fontweight': 'medium'})
        axs[0].legend(['anchor', 'test'])
        axs[0].axhline(min_int, linestyle='dashed', color='red')
        axs[0].axhline(max_int, linestyle='dashed', color='red')

        _ = axs[1].plot(y1, x1, 'o:', label="anchor (measured)")
        _ = axs[1].plot(y2, x2, 'o:', label="test (measured)")
        _ = axs[1].plot(v1, samples, '-', label="anchor (interpolated)")
        _ = axs[1].plot(v2, samples, '-', label="test (interpolated)")
        
        axs[1].legend()
        axs[1].set_xlabel('bitrate (log)', fontsize=21)
        axs[1].set_ylabel('dist', fontsize=21)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=21)
        axs[1].set_title(f'BD rate: {avg_diff:.3f}', fontdict={'fontsize': 24, 'fontweight': 'medium'})
        axs[1].axhline(min_int, linestyle='dashed', color='red')
        axs[1].axhline(max_int, linestyle='dashed', color='red')

        if title and title != "":
            fig.suptitle(title, fontsize=16)

    except ValueError as ve:
        print(ve)
    
    return fig, avg_diff, R1, PSNR1, R2, PSNR2


#####################################################################################################



def main():

    assert len(sys.argv) == 3

    ref = Path(sys.argv[1])
    test = Path(sys.argv[2])

    print("Anchor:", ref)
    print("Test:", test)
    
    metrics = [ m.key for m in (
            Metric.PSNR_Y, 
            Metric.PSNR, 
            Metric.MSSSIM, 
            Metric.VMAF 
        )]

    if ref.is_dir() and (ref.parent / 'streams.csv').exists() \
        and test.is_dir() and (test.parent / 'streams.csv').exists():
        # eg. compare.py ./scenario/codec1/a_key1 ./scenario/codec2/a_key2
        plots = [ m.key for m in (
            Metric.PSNR_Y, 
            Metric.PSNR, 
            Metric.MSSSIM, 
            Metric.VMAF
        )]
        seqid, r = compare_anchors_directories(ref, test, metrics, strict=True, sanitize=True, plots=plots)
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