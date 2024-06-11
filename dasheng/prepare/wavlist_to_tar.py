from functools import partial
from typing import Any, Dict, Iterable
import json
from pathlib import Path
import pandas as pd
import argparse
import multiprocessing
from webdataset import TarWriter
from tqdm import tqdm


def proxy_read(data: Dict, filename_column: str):
    filename = data.pop(filename_column)
    with open(filename, 'rb') as buf:
        raw_data = buf.read()
    fpath = Path(filename)
    stem_name = str(fpath.stem).replace('.', '_')
    suffix = fpath.suffix.replace('.', '')
    ret_data = {
        suffix: raw_data,
        '__key__': f"{stem_name}",  # Just cast to str
    }
    # If we have some labels, also dump a .json file
    if len(data) > 0:
        ret_data['json'] = json.dumps(data).encode('utf-8')
    return ret_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_filelist',
        type=Path,
        help=
        "Some input filelist. We will expect a column named <filename> for the data and every other column will be dumped to json format. The filenames (basepath) need to be unique. Please first shuffle the list before processing."
    )
    parser.add_argument('outputdir', type=Path)
    parser.add_argument('-s', '--size_per_file', type=int, default=10000)
    parser.add_argument('-n', '--n_workers', type=int, default=4)
    parser.add_argument(
        '--filename_column',
        default='filename',
        type=str,
        help="The column name that identifies the files to extract")
    parser.add_argument('-d', '--delim', default='\t', type=str)
    parser.add_argument('--compress',
                        action='store_true',
                        default=False,
                        help="Using tar.gz instead of .tar")
    parser.add_argument(
        '--write_json',
        default=None,
        type=str,
        help=
        "Also writes a json to the target directory. Useful with the 'wids' library to read in random."
    )
    parser.set_defaults(stereo=False)
    args = parser.parse_args()
    df_iterator: Iterable[pd.DataFrame] = pd.read_csv(
        args.input_filelist, sep=args.delim, chunksize=args.size_per_file)

    shards_base_path = args.outputdir
    shards_base_path.mkdir(parents=True, exist_ok=True)

    suffix = '.tar' if args.compress is False else '.tar.gz'

    output_json: Dict[str, Any] = dict(wids_version=1)
    tar_file_outputs = []
    with multiprocessing.Pool(processes=args.n_workers) as pool:
        for file_num, df in enumerate(
                tqdm(df_iterator,
                     leave=True,
                     desc='Dumping to file',
                     unit='shard')):
            #Locally sample
            data = df.sample(frac=1.0).to_dict('records')
            output_file_iter = str(
                shards_base_path /
                f'{args.input_filelist.stem}_{file_num:07d}{suffix}')
            n_samples = len(data)
            tar_file_outputs.append(
                dict(url=str(output_file_iter), nsamples=n_samples))
            with TarWriter(output_file_iter,
                           encoder=False,
                           compress=args.compress) as dst:
                for return_values in tqdm(pool.imap_unordered(
                        partial(proxy_read,
                                filename_column=args.filename_column), data),
                                          unit='file',
                                          total=len(data),
                                          leave=False):
                    dst.write(return_values)
    print(f"Finished, final data can be found at {args.outputdir}")
    if args.write_json is not None:
        import json
        output_json['shardlist'] = tar_file_outputs
        with open(args.write_json, 'w') as f:
            json.dump(output_json, f)
        print(f"Dumped Json for wids usage at {args.write_json}")
