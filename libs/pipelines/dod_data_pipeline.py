from typing import Tuple
from dataclasses import dataclass
import io
import logging
import pandas as pd
from libs.enums import Intervention
from libs import validate_results
from libs import build_dod_dataset
from libs.functions import generate_shapefiles
_logger = logging.getLogger(__name__)


@dataclass
class DodInterventionResult(object):
    key: str
    projection_df: pd.DataFrame
    shapefiles: Tuple[io.BytesIO, io.BytesIO, io.BytesIO]



def run_intervention(
    intervention: Intervention, input_file: str, run_validation: bool = True
):
    states_key_name = f"states.{intervention.name}"
    counties_key_name = f"counties.{intervention.name}"
    # States
    states_df = build_dod_dataset.get_usa_by_states_df(input_file, intervention.value)
    if run_validation:
        validate_results.validate_states_df(states_key_name, states_df)

    state_shapefiles = build_dod_dataset.get_usa_state_shapefile(states_df)
    if run_validation:
        validate_results.validate_states_shapefile(
            states_key_name, *state_shapefiles
        )

    _logger.info(f"Generated state shape files for {intervention.name}")

    counties_df = generate_shapefiles.get_usa_by_county_with_projection_df(
        input_file, intervention.value
    )
    if run_validation:
        validate_results.validate_counties_df(counties_key_name, counties_df)

    county_shapefiles = generate_shapefiles.get_usa_county_shapefile(
        counties_df
    )
    if run_validation:
        validate_results.validate_counties_shapefile(
            counties_key_name, *county_shapefiles
        )

    state_result = DodInterventionResult(
        key=states_key_name, projection_df=states_df, shapefiles=state_shapefiles
    )
    county_result = DodInterventionResult(
        key=counties_key_name, projection_df=counties_df, shapefiles=county_shapefiles
    )
    return state_result, county_result


def deploy_results(results: DodInterventionResult):
    # formats + uploads csv (possibly based on api somewhere)
    # uploads shapefile
    pass
