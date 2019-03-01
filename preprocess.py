import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import vc
from hparams import hparams
from functools import partial
import numpy as np
from util import audio


def preprocess_vc(args):
  metas = []
  with open(os.path.join(args.data_dir, 'metadata.csv')) as f:
      for line in f:
          metas.append(line.strip('\n').strip().split('|'))

  in_dir = os.path.join(args.base_dir, args.data_dir, 'wavs')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = vc.build_from_path(metas, in_dir, out_dir, args.num_workers, tqdm=tqdm)
  print('len(metadata)',len(metadata))
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')

  frames = sum([m[1] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  #print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max([m[1] for m in metadata]))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('.'))
  parser.add_argument('--data_dir', default=os.path.expanduser('parallel_data'))
  parser.add_argument('--output', default='training')
  # parser.add_argument('--dataset', required=True, choices=['mg', 'we'])
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  preprocess_vc(args)
 

if __name__ == "__main__":
  main()
