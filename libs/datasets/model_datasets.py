from typing import Dict, Type, List, NewType
import functools
import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.sources.pyseir_output import PyseirOutput


@functools.lru_cache(20)
def load_models(input_dir, intervention, filters=None):
    data_source = PyseirOutput.load(input_dir, intervention)
    timeseries = TimeseriesDataset.build_from_data_source(
        data_source, fill_missing_state=False
    )
    for data_filter in filters or []:
        timeseries = data_filter.apply(timeseries)

    return timeseries
