import click
import csv
from pathlib import Path
from typing import Iterable, List, Tuple, Any
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.figure import Figure

from constants import BITSTREAMS_DIR, SEQUENCES_DIR
from anchor import VariantData, VariantMetricSet, Metric, load_variants, AnchorTuple
from metrics import load_csv_metrics


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


def sanitize_rd_data(rates, dist, step=0.001):
    """workaround for saturated dist values.
        returns sanitized rates & dist, sorted on dist.
        consecutive samples that have the same values are modified with the given step
        so that the sequence is increasing instead of stagnating.
        if a sequence is decreasing rather than stagnating, it is not modified.
        eg. [50., 50., 50., 50., 50.] becomes [50., 50.001, 50.002, 50.003, 50.004]
            [98.999, 99.999, 99.999, 99.999, 100.] becomes [98.999, 99.999, 100., 100.001, 100.002]
            [98.999, 99.999, 99.999, 97., 100.] becomes [97, 98., 99.999, 100., 100.001]
        """
    rate = np.array(rates)
    dist = np.array(dist)
    sorted = np.lexsort((rate, dist))
    rate = rate[sorted]
    dist = dist[sorted]
    dist_fix = []
    sanitized = False
    for i, _ in enumerate(rate):
        d = dist[i]
        if i and (d == dist[i - 1]):
            d = dist_fix[-1] + step
            sanitized = True
        elif i and (d > dist[i - 1]) and (d <= dist_fix[-1]):
            d = dist_fix[-1] + step
            sanitized = True
        dist_fix.append(d)
    if sanitized:
        print("/!\\ data has been sanitized:")
        print(f" -  replaced: {dist}")
        print(f" -  with    : {dist_fix}")
    return rate, np.array(dist_fix, dtype=np.float64)


def rd_metrics(variants: List[VariantData], rate:Metric, dist:Metric) -> Iterable[Any]:
    # return zip(*[(v.metrics[rate], v.metrics[dist]) for v in variants])
    rd = []
    for v in variants:
        r = v.metrics[rate]
        if dist in v.metrics:
            rd.append((r, v.metrics[dist]))
        else:
            rd.append((r, 0.000))
    return zip(*rd)


def compare_anchors_metrics(anchor: List[VariantData], test: List[VariantData], dist:Metric, rate=Metric.BITRATE, anchor_label="anchor", test_label="test", sanitize=True) -> Tuple[Figure, int, Any, Any, Any, Any]:
    anchor_metrics = [*rd_metrics(anchor, rate=rate, dist=dist)]
    test_metrics = [*rd_metrics(test, rate=rate, dist=dist)]
    q_key = dist.csv_key.replace('_', ' ')
    return bd_rate_plot(*anchor_metrics, *test_metrics, sanitize=sanitize, anchor_label=anchor_label, test_label=test_label, quality_label=f'Quality ({q_key})')


def compare_sequences(anchor: AnchorTuple, test: AnchorTuple, metrics: Iterable[Metric], sanitize=True, save_plots=False, strict=False) -> Iterable[Tuple[str, VariantMetricSet]]:

    anchor_variants = [v for (_, v) in load_variants(anchor)]
    load_csv_metrics(anchor, anchor_variants)

    test_variants = [v for (_, v) in load_variants(test)]
    load_csv_metrics(test, test_variants)
    
    res = {}
    fig = None
    for m in metrics:
        try:
            fig, bd, *_ = compare_anchors_metrics(anchor_variants, test_variants, dist=m, anchor_label=str(anchor.anchor_key), test_label=str(test.anchor_key), sanitize=sanitize)
            plt.ioff()
            plt.close(fig)
            res[m] = f'{round(bd, 3):.3f}'
            if save_plots:
                fname = test.working_dir / 'Characterization' / f'{anchor.working_dir.name}.{test.working_dir.name}.{m}.png'.lower()
                if not fname.parent.exists():
                    fname.parent.mkdir()
                fig.savefig(fname)
                plt.close(fig)

        except BaseException as e:
            if fig:
                plt.close(fig)
            if strict:
                raise
            print(e)
            res[m] = e

    return anchor.reference.sequence['Key'], res


def compare_anchors(anchors: List[AnchorTuple], tests: List[AnchorTuple], metrics: List[str], save_plots=False, strict=False):

    assert len(anchors) == len(tests), f'reference has {len(anchors)} anchors, test has {len(tests)} anchors.'

    bd_rates = []

    for aref, atest in zip(anchors, tests):

        vref = [v for (_, v) in load_variants(aref)]
        load_csv_metrics(aref, vref)

        vtest = [v for (_, v) in load_variants(atest)]
        load_csv_metrics(atest, vtest)

        arates = {}
        for key in metrics:
            try:
                fig, bd, *_ = compare_anchors_metrics(vref, vtest, dist=key, anchor_label=str(aref.anchor_key), test_label=str(atest.anchor_key))
                plt.ioff()
                plt.close(fig)
                arates[key] = f'{round(bd, 3):.3f}'
                if save_plots:
                    fname = atest.working_dir.parent / 'Characterization' / (f'{aref.working_dir.name}.{atest.working_dir.name}'.upper() + f'{key}.png'.lower())
                    if not fname.parent.exists():
                        fname.parent.mkdir()
                    fig.savefig(fname)
                    try:
                        plt.close()
                    except BaseException as e:
                        print(e)
                    
            except BaseException as e:
                if strict:
                    raise
                arates[key] = e

        bd_rates.append((aref, atest, arates))

    return [(r.reference.sequence['Key'], bdr) for (r, _, bdr) in bd_rates]


def csv_dump(data, fp, metrics, extras=['reference']):
    stats = {
        'min': {},
        'max': {},
        'avg': {},
    }

    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    with open(fp, 'w', newline='') as f:
        
        fieldnames = extras + [k.csv_key for k in metrics]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in data:

            row = {}

            for k in extras:
                row[k] = r[k]

            for k in metrics:
                row[k.csv_key] = r[k]

                if len(data) == 1:
                    continue
                if k == "reference":
                    continue

                v = row[k.csv_key]
                if isinstance(v, BaseException):
                    print(v)
                    continue
                try:
                    v = float(v)
                except BaseException as e:
                    print(e)
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

        r = {"reference": "Min"}
        for k in metrics:
            r[k.csv_key] = stats["min"].get(k, None)
        writer.writerow(r)

        r = {"reference": "Max"}
        for k in metrics:
            r[k.csv_key] = stats["max"].get(k, None)
        writer.writerow(r)

        r = {"reference": "Avg"}
        for k in metrics:
            avg = None
            if k in stats["avg"] and len(stats["avg"][k]):
                avg = 0
                for j in stats["avg"][k]:
                    avg += j
                avg /= len(stats["avg"][k])
            r[k.csv_key] = avg
        writer.writerow(r)


def strictly_increasing(samples):
    for i, v in enumerate(samples):
        if i and v <= samples[i - 1]:
            return False
    return True


def bd_rate_plot(R1, DIST1, R2, DIST2, anchor_label="anchor", test_label="test", quality_label="Quality", bitrate_label = "Bit rate in bit/s", sanitize=False, debug=False) -> Tuple[Figure, int, Any, Any, Any, Any]:
    """adapted from https://github.com/Anserw/Bjontegaard_metric
    which computes bd-rate according to:
        [1] G. Bjontegaard, Calculation of average PSNR differences between RD-curves (VCEG-M33)
        [2] S. Pateux, J. Jung, An excel add-in for computing Bjontegaard metric and its evolution
    """
    if sanitize:
        R1, DIST1 = sanitize_rd_data(R1, DIST1)
        R2, DIST2 = sanitize_rd_data(R2, DIST2)
        _ = strictly_increasing(DIST1) and strictly_increasing(DIST2)

    else:
        DIST1 = np.array(DIST1)
        DIST2 = np.array(DIST2)

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # integration interval
    min_int = max(min(DIST1), min(DIST2))
    max_int = min(max(DIST1), max(DIST2))

    samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
    [r1, d1] = sort_on_rates(lR1, DIST1)
    assert strictly_increasing(d1), f'{anchor_label}: {quality_label} not strictly increasing: {d1}'

    [r2, d2] = sort_on_rates(lR2, DIST2)
    assert strictly_increasing(d2), f'{test_label}: {quality_label} not strictly increasing: {d2}'

    v1, v2, avg_diff, fig = None, None, 0, None
    try:
        v1 = scipy.interpolate.pchip_interpolate(d1, r1, samples)
        v2 = scipy.interpolate.pchip_interpolate(d2, r2, samples)

        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (np.exp(avg_exp_diff) - 1) * -100

        nplots = 3 if debug else 2
        fig, axs = plt.subplots(1, nplots, figsize=(30, 12))

        axs[0].plot(R1, DIST1, 'o-', R2, DIST2, 'o-')
        axs[0].set_xlabel(bitrate_label, fontsize=21)
        axs[0].set_ylabel(quality_label, fontsize=21)
        axs[0].grid(True)
        axs[0].tick_params(axis='both', which='major', labelsize=21)
        axs[0].set_title('Rate-Quality Curve', fontdict={'fontsize': 21, 'fontweight': 'medium'})
        axs[0].legend([anchor_label, test_label])
        axs[0].axhline(min_int, linestyle='dashed', color='red')
        axs[0].axhline(max_int, linestyle='dashed', color='red')

        axs[1].plot(r1, d1, 'o:', label=anchor_label)
        axs[1].plot(r2, d2, 'o:', label=test_label)
        axs[1].plot(v1, samples, '-') #, label="anchor (interpolated)")
        axs[1].plot(v2, samples, '-') #, label="test (interpolated)")

        axs[1].legend()
        axs[1].set_xlabel(bitrate_label, fontsize=21)
        axs[1].set_ylabel(quality_label, fontsize=21)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=21)
        axs[1].set_title(f'BD rate gain: {avg_diff:.3f}', fontdict={'fontsize': 21, 'fontweight': 'medium'})
        axs[1].axhline(min_int, linestyle='dashed', color='red')
        axs[1].axhline(max_int, linestyle='dashed', color='red')
        axs[1].fill_betweenx(samples, v1, v2, color='red', alpha=0.25)

        if debug:
            x = [i * interval for i in range(100)]
            axs[2].set_xlabel('composite trapezoidal integration', fontsize=21)
            axs[2].plot(x, v1, '-', label="anchor (interpolated)")
            axs[2].plot(x, v2, '-', label="test (interpolated)")
            axs[2].fill_between(x, v1, v2, color='red', alpha=0.75)

    except ValueError as ve:
        print(ve)
        print('d1: ', d1)
        print('d2: ', d2)

    return fig, avg_diff, R1, DIST1, R2, DIST2








@click.command()
@click.option('--working-dir', envvar='VCC_WORKING_DIR', required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True, readable=True),
    help="directory containing bitstreams and pre-computed metrics, can be set with VCC_WORKING_DIR environment variable." )
@click.option('-p', '--plot', required=False, default=False, is_flag=True, help="enable rd curve & bd rate plots when comparing encoder configs.")
@click.option('-s/-c', required=True, default=True, help="specifies if anchor/test are: sequence IDs or encoder configs IDs")
@click.argument('anchor_key', required=True)
@click.argument('test_key', required=True)
@click.argument('metric_keys', nargs=-1, required=False)
def main(working_dir:str, plot:bool, s:bool, anchor_key:str, test_key:str, metric_keys:Tuple[str]):
    """
    The script expects data to follow data organization as found in: https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/
    \b

    \b
    Comparing anchors using sequence IDs S1-A01-264 to S1-T01-ETM:

    \b
        compare.py -s S1-A01-264 S1-T01-ETM
  
    \b
        plots will be found in S1/ETM/S1-T01-ETM/Characterization

    \b
    Comparing encoder config 'S1-JM-01' to 'S1-HM-01':

    \b
        compare.py -p -c S1-JM-01 S1-HM-01
    
    \b
        results will be found in S1/265/Characterization
    
    """
    root_dir = Path(working_dir)
    bitstreams_dir = root_dir / BITSTREAMS_DIR
    sequences_dir = root_dir / SEQUENCES_DIR
    
    
    # don't process/plot these metrics
    METRIC_BLACKLIST = ( 
        Metric.BITRATE, 
        Metric.BITRATELOG, 
        Metric.ENCODETIME,
        Metric.DECODETIME 
    )

    metrics = []

    for k in metric_keys:
        m = Metric.from_csv_key(k)
        if m == None:
            click.echo(f'Aborting - invalid metric key specified: "{k}"')
            return
        else:
            metrics.append(m)
    
    if s:
        anchor = AnchorTuple.load(anchor_key, bitstreams_dir, sequences_dir)
        test = AnchorTuple.load(test_key, bitstreams_dir, sequences_dir)
        if len(metrics) == 0:
            metrics = [m for m in anchor.get_metrics_set() if m not in METRIC_BLACKLIST]
        _ = compare_sequences(anchor, test, metrics, save_plots=True)

    else:
        anchors = AnchorTuple.iter_cfg_anchors(anchor_key, bitstreams_dir, sequences_dir)
        tests = AnchorTuple.iter_cfg_anchors(test_key, bitstreams_dir, sequences_dir)
        t = tests[-1]
        data = []
        if len(metrics) == 0:
            metrics = [m for m in t.get_metrics_set() if m not in METRIC_BLACKLIST]
        for (seqid, r) in compare_anchors(anchors, tests, metrics, save_plots=True):
            r['reference'] = seqid
            data.append(r)
        outp = t.working_dir.parent  / 'Characterization' / f'{anchor_key}.{test_key}.csv'.lower()
        csv_dump(data, outp, metrics)

if __name__ == "__main__":
    main()
