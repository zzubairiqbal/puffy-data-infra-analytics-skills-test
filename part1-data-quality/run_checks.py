import argparse
from dq_framework.runner import run_checks_on_folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing raw daily event CSV exports")
    ap.add_argument("--output-dir", required=True, help="Folder to write reports")
    args = ap.parse_args()
    run_checks_on_folder(args.input_dir, args.output_dir)
    print(f"Wrote reports to {args.output_dir}")

if __name__ == "__main__":
    main()
