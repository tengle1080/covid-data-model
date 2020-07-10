from typing import Type, Tuple
import os
import tempfile
import pathlib
import datetime
import enum
import git
from urllib.parse import urlparse

import boto3
import structlog
import numpy as np
import pandas as pd
import pydantic
from covidactnow.datapublic import common_df

from libs.datasets import dataset_base
from libs.datasets import combined_datasets
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets import dataset_utils

_logger = structlog.getLogger(__name__)


REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

DATA_CACHE_FOLDER = REPO_ROOT / ".data"


def s3_split(url) -> Tuple[str, str]:
    """Split S3 URL into bucket and key.

    Args:
        url: S3 URL.

    Returns: Tuple of bucket, key
    """
    results = urlparse(url, allow_fragments=False)
    return results.netloc, results.path.lstrip("/")


class DatasetType(enum.Enum):

    TIMESERIES = "timeseries"
    LATEST = "latest"

    @property
    def dataset_class(self) -> Type:
        """Returns associated dataset class."""
        if self is DatasetType.TIMESERIES:
            return timeseries.TimeseriesDataset

        if self is DatasetType.LATEST:
            return latest_values_dataset.LatestValuesDataset


class DatasetPromotion(enum.Enum):
    """"""

    # Latest dataset
    LATEST = "latest"

    # Has gone through validation
    STABLE = "stable"


class GitSummary(pydantic.BaseModel):

    sha: str
    branch: str
    is_dirty: bool

    @classmethod
    def from_repo_path(cls, path: pathlib.Path):
        repo = git.Repo(path)

        return cls(sha=repo.head.commit.hexsha, branch=str(repo.head.ref), is_dirty=repo.is_dirty())


class CombinedDatasetPointer(pydantic.BaseModel):
    """Describes a persisted combined dataset."""

    dataset_type: DatasetType

    s3_path: str

    # Sha of covid-data-public for dataset
    data_git_info: GitSummary

    # Sha of covid-data-model used to create dataset.
    model_git_info: GitSummary

    # When local file was saved.
    updated_at: datetime.datetime

    @property
    def filename(self) -> str:
        *_, filename = os.path.split(self.s3_path)
        return filename

    def download(
        self, directory: pathlib.Path = DATA_CACHE_FOLDER, overwrite=False, s3_client=None
    ) -> pathlib.Path:
        s3_client = s3_client or boto3.client("s3")
        dest_path = directory / self.filename
        if dest_path.exists() and not overwrite:
            raise FileExistsError()

        bucket, key = s3_split(self.s3_path)
        s3_client.download_file(bucket, key, str(dest_path))
        return dest_path

    def upload_dataset(self, dataset, s3_client):
        with tempfile.NamedTemporaryFile() as tmp_file:
            path = pathlib.Path(tmp_file.name)
            dataset.to_csv(path)
            bucket, key = s3_split(self.s3_path)
            s3_client.upload_file(str(path), bucket, key)
            _logger.info("Successfully uploaded dataset", s3_path=self.s3_path)

    def load_dataset(self, download_directory: pathlib.Path = DATA_CACHE_FOLDER):
        path = download_directory / self.filename
        if not path.exists():
            path = self.download(directory=download_directory)

        return self.dataset_type.dataset_class.load_csv(path)

    def save(self, directory: pathlib.Path, promotion_level: DatasetPromotion):
        filename = form_pointer_filename(self.dataset_type, promotion_level)
        path = directory / filename
        path.write_text(self.json(indent=2))
        return path


def _form_filename(
    dataset_type: DatasetType, data_git_info: GitSummary, model_git_info: GitSummary
) -> str:
    path_format = "{dataset_type}.{timestamp}.{model_sha}-{data_sha}.csv"
    return path_format.format(
        dataset_type=dataset_type.value,
        data_sha=data_git_info.sha[:8],
        model_sha=model_git_info.sha[:8],
        timestamp=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
    )


def form_pointer_filename(dataset_type: DatasetType, promotion_level: DatasetPromotion) -> str:
    return f"{dataset_type.value}.{promotion_level.value}.json"


def persist_dataset(
    dataset: dataset_base.DatasetBase,
    s3_path_prefix: str,
    data_public_path: pathlib.Path = dataset_utils.LOCAL_PUBLIC_DATA_PATH,
    s3_client=None,
) -> CombinedDatasetPointer:
    s3_client = s3_client or boto3.client("s3")

    model_git_info = GitSummary.from_repo_path(REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    if isinstance(dataset, timeseries.TimeseriesDataset):
        dataset_type = DatasetType.TIMESERIES
    elif isinstance(dataset, latest_values_dataset.LatestValuesDataset):
        dataset_type = DatasetType.LATEST

    filename = _form_filename(dataset_type, data_git_info, model_git_info)
    s3_dataset_path = os.path.join(s3_path_prefix, filename)
    dataset_pointer = CombinedDatasetPointer(
        dataset_type=dataset_type,
        s3_path=s3_dataset_path,
        data_git_info=data_git_info,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    dataset_pointer.upload_dataset(dataset, s3_client)
    return dataset_pointer


def update_data_public_head(
    s3_bucket: str,
    pointer_path_dir: pathlib.Path = REPO_ROOT,
    latest_dataset=None,
    timeseries_dataset=None,
):

    s3_path_prefix = f"s3://{s3_bucket}"
    if not latest_dataset:
        latest_dataset = combined_datasets.build_us_latest_with_all_fields(skip_cache=True)
    latest_pointer = persist_dataset(latest_dataset, s3_path_prefix)
    latest_pointer.save(pointer_path_dir, DatasetPromotion.LATEST)

    if not timeseries_dataset:
        timseries_dataset = combined_datasets.build_timeseries_with_all_fields(skip_cache=True)
    timeseries_pointer = persist_dataset(timeseries_dataset, s3_path_prefix)
    timeseries_pointer.save(pointer_path_dir, DatasetPromotion.LATEST)
    return latest_pointer, timeseries_pointer


def load_us_timeseries_with_all_fields(
    promotion_level: DatasetPromotion = DatasetPromotion.LATEST,
    pointer_directory: pathlib.Path = REPO_ROOT,
    dataset_download_directory: pathlib.Path = DATA_CACHE_FOLDER,
) -> timeseries.TimeseriesDataset:
    filename = form_pointer_filename(DatasetType.TIMESERIES, promotion_level)
    pointer_path = pointer_directory / filename
    pointer = CombinedDatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(download_directory=dataset_download_directory)


def load_us_latest_with_all_fields(
    promotion_level: DatasetPromotion = DatasetPromotion.LATEST,
    pointer_directory: pathlib.Path = REPO_ROOT,
    dataset_download_directory: pathlib.Path = DATA_CACHE_FOLDER,
) -> latest_values_dataset.LatestValuesDataset:
    filename = form_pointer_filename(DatasetType.LATEST, promotion_level)
    pointer_path = pointer_directory / filename
    pointer = CombinedDatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(download_directory=dataset_download_directory)
