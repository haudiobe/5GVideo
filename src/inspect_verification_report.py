import csv
import sys
import ast
from pathlib import Path


def find_errors_in_report(csv_report:Path, threshold=0.01):

    def parse_info(info):
        l = ast.literal_eval(info)
        return {k: v for (k, v) in l}

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


def help():
    print('inspect_verification_report.py verification.csv [threshold]')


def main():
    if sys.argv == 1:
        help()
    csv_report = Path(sys.argv[1])
    if not csv_report.is_file():
        print('FILE NOT FOUND: ', csv_report)
        help()
        return
    try:
        threshold = float(sys.argv[2]) if len(sys.argv) == 3 else 0.01
    except BaseException as e:
        print(e)
        help()
        return

    errors = find_errors_in_report(csv_report, threshold)
    if len(errors):
        for variant_key, details in errors.items():
            print("#", variant_key)
            for k, v in details.items():
                print(f'{k} - found: {v[0]} - expected: {v[1]}')
        print("\n")
    
if __name__  == "__main__":
    main()