from typing import List
import pathlib
from collections import defaultdict
import logging
import pydantic
import simplejson
from api.can_api_definition import CountyFipsSummary
from api.can_api_definition import CovidActNowCountyTimeseries
from api.can_api_definition import PredictionTimeseriesRowWithHeader
from libs.datasets import combined_datasets
from libs.datasets import model_datasets
from libs.enums import Intervention
from libs import dataset_deployer

from libs.functions import generate_api as api

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


def run_for_intervention(aggregate_level, intervention, filters):

    logger.info(f"Running intervention {intervention.name}")

    # Loading data and applying filters
    historical = combined_datasets.build_timeseries_with_all_fields(
        filters=filters
    )
    latest_values = combined_datasets.build_latest_in_context_with_all_fields(
        filters=filters
    )
    model_timeseries = model_datasets.load_models(
        intervention, filters=filters
    )
    projections = calculate_projections(model_timeseries)

    # Build API Output building blocks
    timeseries = build_timeseries(
        historical, latest_values, projections, model_timeseries
    )
    summaries = build_summaries(projections, latest_values)

    # Upload data
    upload_timeseries(aggregate_level, timeseries)
    upload_summary(aggregate_level, summaries)
    upload_bulk_summary(aggregate_level, timeseries)
    upload_bulk_timeseries(aggregate_level, summaries)


def build_summaries(projection_dataset, latest_values) -> List[CovidActNowSummary]:

    summaries = []
    for fips in projection_dataset.data.fips.unique():
        latest = latest_values.get_data_for_fips(fips)
        projection_data = projection_dataset.get_data_for_fips(fips)
        summary = api.generate_summary(latest, projection_data)
        summaries.append(summary)

    return summaries


def build_timeseries(
    historical, latest_values, projection_dataset, model_timeseries
) -> List[CovidActNowTimeseries]:

    timeseries = []
    for fips in projection_dataset.data.fips.unique():
        latest_data = latest_values.get_data_for_fips(fips)
        projection_data = projection_dataset.get_data_for_fips(fips)
        model_data = model_timeseries.get_data_for_fips(fips)
        historical_data = historical.get_data_for_fips(fips)

        summary = api.generate_timeseries(
            projection_data,
            model_data,
            historical_data,
            latest_data
        )
        timeseries.append(summary)

    return timeseries


def build_county_summary_from_model_output(input_dir) -> List[APIOutput]:
    """Builds lists of counties available from model output.

    Args:
        input_dir: Input directory.  Should point to county output.

    Returns: List of API Output objects.
    """
    found_fips_by_state = defaultdict(set)
    found_fips = set()
    for path in pathlib.Path(input_dir).iterdir():
        if not str(path).endswith(".json"):
            continue

        state, fips, intervention, _ = path.name.split(".")
        found_fips_by_state[state].add(fips)
        found_fips.add(fips)

    results = []
    for state, data in found_fips_by_state.items():
        fips_summary = CountyFipsSummary(counties_with_data=sorted(list(data)))
        output = APIOutput(f"{state}.summary", fips_summary, None)
        results.append(output)

    fips_summary = CountyFipsSummary(counties_with_data=sorted(list(found_fips)))
    output = APIOutput(f"fips_summary", fips_summary, None)
    results.append(output)
    return results


def remove_root_wrapper(obj: dict):
    """Removes __root__ and replaces with __root__ value.

    When pydantic models are used to wrap lists this is done using a property __root__.
    When this is serialized using `model.json()`, it will return a json list. However,
    calling `model.dict()` will return a dictionary with a single key `__root__`.
    This function removes that __root__ key (and all sub pydantic models with a
    similar structure) to have a similar hierarchy to the json output.

    A dictionary {"__root__": []} will return [].

    Args:
        obj: pydantic model as dict.

    Returns: object with __root__ removed.
    """
    # Objects with __root__ should have it as the only key.
    if len(obj) == 1 and '__root__' in obj:
        return obj['__root__']

    results = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            value = remove_root_wrapper(value)

        results[key] = value

    return results


def deploy_results(results, output: str, write_csv=False):
    """Deploys results from the top counties to specified output directory.

    Args:
        result: Top Counties Pipeline result.
        key: Name for the file to be uploaded
        output: output folder to save results in.
    """
    output_path = pathlib.Path(output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for api_row in results:
        data = remove_root_wrapper(api_row.data.dict())
        # Encoding approach based on Pydantic's implementation of .json():
        # https://github.com/samuelcolvin/pydantic/pull/210/files
        data_as_json = simplejson.dumps(
            data, ignore_nan=True, default=pydantic.json.pydantic_encoder
        )
        dataset_deployer.upload_json(api_row.file_stem, data_as_json, output)
        if write_csv:
            if not isinstance(data, list):
                raise ValueError("Cannot find list data for csv export.")
            dataset_deployer.write_nested_csv(data, api_row.file_stem, output)


def build_prediction_header_timeseries_data(data):

    rows = []
    api_data = data.data
    # Iterate through each state or county in data, adding summary data to each
    # timeseries row.
    for row in api_data.__root__:
        county_name = None
        if isinstance(row, CovidActNowCountyTimeseries):
            county_name = row.countyName

        summary_data = {
            "countryName": row.countryName,
            "countyName": county_name,
            "stateName": row.stateName,
            "fips": row.fips,
            "lat": row.lat,
            "long": row.long,
            "intervention": data.intervention.name,
            "lastUpdatedDate": row.lastUpdatedDate,
        }

        for timeseries_data in row.timeseries:
            timeseries_row = PredictionTimeseriesRowWithHeader(
                **summary_data, **timeseries_data.dict()
            )
            rows.append(timeseries_row)

    return APIOutput(data.file_stem, rows, data.intervention)


def deploy_prediction_timeseries_csvs(data: APIOutput, output):
    dataset_deployer.write_nested_csv(
        [row.dict() for row in data.data], data.file_stem, output
    )
