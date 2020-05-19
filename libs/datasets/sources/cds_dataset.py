import logging
import numpy
import pandas as pd
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields

_logger = logging.getLogger(__name__)


def fill_missing_county_with_city(row):
    """Fills in missing county data with city if available.

    """
    if pd.isnull(row.county) and not pd.isnull(row.city):
        if row.city == "New York City":
            return "New York"
        return row.city

    return row.county


class CDSDataset(data_source.DataSource):
    DATA_PATH = "data/cases-cds/timeseries.csv"
    SOURCE_NAME = "CDS"

    class Fields(object):
        CITY = "city"
        COUNTY = "county"
        STATE = "state"
        COUNTRY = "country"
        POPULATION = "population"
        LATITUDE = "lat"
        LONGITUDE = "long"
        LEVEL = "level"
        URL = "url"
        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"
        ACTIVE = "active"
        TESTED = "tested"
        GROWTH_FACTOR = "growthFactor"
        DATE = "date"
        AGGREGATE_LEVEL = "aggregate_level"
        FIPS = "fips"
        NEGATIVE_TESTS = "negative_tests"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CASES: Fields.CASES,
        CommonFields.POSITIVE_TESTS: Fields.CASES,
        CommonFields.NEGATIVE_TESTS: Fields.NEGATIVE_TESTS,
        CommonFields.POPULATION: Fields.POPULATION,
    }

    TEST_FIELDS = [
        Fields.COUNTRY,
        Fields.STATE,
        Fields.COUNTY,
        Fields.FIPS,
        Fields.DATE,
        Fields.CASES,
        Fields.TESTED,
    ]

    @classmethod
    def local(cls) -> "CDSDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(input_path, parse_dates=[cls.Fields.DATE], dtype={"fips": str})
        data = cls.standardize_data(data)
        return cls(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        city_data = data["level"] == "city"
        data = data.loc[~city_data, :]

        data[cls.Fields.AGGREGATE_LEVEL] = data[cls.Fields.LEVEL]
        data[cls.Fields.NEGATIVE_TESTS] = data[cls.Fields.TESTED] - data[cls.Fields.CASES]
        # There are a handful of regions that have
        combined_fips = data.fips.apply(lambda x: len(x) if not pd.isna(x) else 0) > 5
        data = data.loc[~combined_fips, :]
        return data

    @classmethod
    def remove_duplicate_city_data(cls, data):
        # Don't want to return city data because it's duplicated in county
        # City data before 3-23 was not duplicated.
        # data = data[data[cls.Fields.CITY].isnull()]
        pre_march_23 = data[data.date < "2020-03-23"]
        pre_march_23.county = pre_march_23.apply(fill_missing_county_with_city, axis=1)
        split_data = [
            pre_march_23,
            data[(data.date >= "2020-03-23") & data[cls.Fields.CITY].isnull()],
        ]
        data = pd.concat(split_data)
        return data
