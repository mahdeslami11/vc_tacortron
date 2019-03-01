from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


def build_from_path(metas, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      metas: list of metadata, like [[source_wav_path, target_wav_path], ...]
      in_dir: The directory where you raw parallel corpus stored
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  for m in metas:
    source_wav_path = os.path.join(in_dir, m[0])
    target_wav_path = os.path.join(in_dir, m[1])
    futures.append(executor.submit(partial(_process_utterance, out_dir, index, source_wav_path, target_wav_path)))
    index += 1

  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, source_wav_path, target_wav_path):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  source_wav = audio.load_wav(source_wav_path) 
  target_wav = audio.load_wav(target_wav_path)

  # Compute the linear-scale spectrogram from the wav:
  target_spectrogram = audio.spectrogram(target_wav).astype(np.float32)
  n_frames = target_spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  source_mel_spectrogram = audio.melspectrogram(source_wav).astype(np.float32)
  target_mel_spectrogram = audio.melspectrogram(target_wav).astype(np.float32)
  
  # Write the spectrograms to disk:
  #source_spectrogram_filename = 'source-spec-%05d.npy' % index
  source_mel_filename = 'source-mel-%05d.npy' % index
  target_spectrogram_filename = 'target-spec-%05d.npy' % index
  target_mel_filename = 'target-mel-%05d.npy' % index
  #np.save(os.path.join(out_dir, source_spectrogram_filename), source_spectrogram.T, allow_pickle=False)
  
  np.save(os.path.join(out_dir, source_mel_filename), source_mel_spectrogram.T, allow_pickle=False) 
  np.save(os.path.join(out_dir, target_spectrogram_filename), target_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, target_mel_filename), target_mel_spectrogram.T, allow_pickle=False)   
 
  # Return a tuple describing this training example:
  return (source_mel_filename, n_frames, target_spectrogram_filename,target_mel_filename )
