#!.venv/bin/python3

import sys

def parse(logf):
    summary = False
    keys = None
    values = None
    for l in logf.readlines():
        if summary:
            if keys:
                if not values:
                    values = l.split()[2:] # '1', 'a'
                # break
            else:
                keys = l.split()[3:] # 'Total', 'Frames', '|'
        else:
            summary = l.startswith('SUMMARY')
        if values:
            if l.startswith(' Total Time:'):
                keys.append('EncodeTime')
                values.append(l.split()[-2])
    assert summary and keys and values
    metrics = {}
    for i, k in enumerate(keys):
        metrics[k] = float(values[i])
    return metrics

def main():
    assert len(sys.argv) == 2, 'bad cmd'
    with open(sys.argv[1], 'r') as logf:
        d = parse(logf)
        print(d)

if __name__ == "__main__":
    main()