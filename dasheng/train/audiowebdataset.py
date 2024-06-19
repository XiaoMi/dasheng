from typing import List, Optional
import random
from functools import partial
import webdataset as wds
import braceexpand # Dependency in wds
import torch
import numpy as np

def crop_or_pad(wav: torch.Tensor, crop_size: int, pad_last: bool = False):
    n_samples, *_ = wav.shape
    available_crops = n_samples // crop_size
    for i in range(available_crops):
        crop = wav[i * crop_size:(i + 1) * crop_size, ...]
        yield crop

    if (available_crops == 0) or (pad_last):
        last_crop = wav[available_crops * crop_size:, ...]
        padded = torch.zeros((crop_size, *last_crop.shape[1:]))
        padded[:last_crop.shape[0]] = last_crop
        yield padded


def convert_decibels_to_amplitude_ratio(decibels):
    return 10**(decibels / 20)


def _audio_gain(data_stream, min_gain_db: float = -6, max_gain_db=10):
    for sample in data_stream:
        audio, *extra = sample
        scale_factor = convert_decibels_to_amplitude_ratio(
            random.uniform(min_gain_db, max_gain_db))
        yield (audio * scale_factor, *extra)

def _seq_crop(data, crop_size: int, mono: bool = True, pad_last: bool = False, drop_crops: bool = False,handler=None):
    """WebDataset crop filter, yields sequential crops"""
    for sample in data:
        audio, *extra = sample
        if isinstance(audio, tuple):
            audio = audio[0]
        if mono and audio.ndim == 2:
            audio = audio.mean(0)
        if drop_crops and audio.shape[-1] < int(crop_size * 0.8):
            continue
        crops = crop_or_pad(audio.float(),
                            crop_size=crop_size,
                            pad_last=pad_last)
        for crop in crops:
            yield (crop, *extra)


class Audiowebdataset_Fluid(wds.DataPipeline):

    def __init__(self,
                 urls,
                 shuffle: Optional[int] = None,
                 crop_size: int = 16000,
                 resample: bool = False,
                 crop_shuffle: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 add_gain: bool = False,
                 drop_crops: bool = False,
                 with_json: bool = False,

                 ):
        pipeline: List = [
            wds.SimpleShardList(urls)
            if resample is False else wds.ResampledShards(urls)
        ]
        if shuffle is not None:
            # Tar wise shuffle
            pipeline.extend([
                wds.detshuffle(
                    bufsize=shuffle,
                    initial=shuffle // 4,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(
                    bufsize=shuffle,
                    initial=shuffle // 4,
                ),
            ])
        else:
            pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples()])
        pipeline.extend([
            wds.decode(wds.torch_audio, handler=wds.warn_and_continue),
            wds.to_tuple("mp3;wav;flac", "json", "__key__") if with_json else wds.to_tuple("mp3;wav;flac", "__key__"),
            partial(_seq_crop, crop_size=crop_size, drop_crops = drop_crops)
        ])
        if add_gain:
            pipeline.extend([_audio_gain])
        if crop_shuffle is not None:
            pipeline.append(wds.shuffle(crop_shuffle))
        if batch_size is not None:
            pipeline.append(wds.batched(batch_size))
        super().__init__(pipeline)


#Can also replace with wds.Randomix
class SampleDatasets(wds.DataPipeline, wds.compat.FluidInterface):

    def __init__(self, datasets, probability: Optional[List[float]] = None):
        super().__init__()
        self.datasets = datasets
        if probability is None:
            probability = [1.0] * len(self.datasets)
        self.prob = probability

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in random.choices(sources, weights=self.prob):
                try:
                    yield next(source)
                except StopIteration:
                    return


def create_dataloader(data_urls: List[str],
                      crop_size: int,
                      batch_size: int = 32,
                      crop_shuffle: Optional[int] = None,
                      resampled: bool = False,
                      num_workers: int = 4,
                      *args,
                      **kwargs):
    train_lists: List[str] = []
    for train_data_url in data_urls:
        train_lists.extend(braceexpand.braceexpand(train_data_url))
    ds = Audiowebdataset_Fluid(
        train_lists,
        crop_size=crop_size,
        resample=resampled,
        batch_size=batch_size,
        crop_shuffle=crop_shuffle,
        shuffle=crop_shuffle,
    )
    dataloader = wds.WebLoader(ds, batch_size=None, num_workers=num_workers)
    if crop_shuffle is not None:
        dataloader = dataloader.unbatched().shuffle(crop_shuffle).batched(
            batch_size)
    return dataloader
