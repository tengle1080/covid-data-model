#!/usr/bin/env python
from io import BytesIO
import boto3
import click
import os
import logging

from libs.enums import Intervention
from libs import validate_results
from libs import build_dod_dataset
from libs.pipelines import dod_data_pipeline
from libs.functions.generate_shapefiles import (
    get_usa_county_shapefile,
    get_usa_state_shapefile,
)

_logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command()
@click.option(
    "--run_validation",
    "-r",
    default=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-file",
    "-i",
    default="results",
    help="Input directory of state/county projections",
)
@click.option(
    "--output", "-o", default="results/dod", help="Output directory for artifacts"
)
def deploy(run_validation, input_file, output):
    """The entry function for invocation"""

    for intervention in list(Intervention):
        _logger.info(f"Starting to generate files for {intervention_enum.name}.")
        state_results, county_results = dod_data_pipeline.run_intervention(
            intervention, input_file, run_validation
        )
        dod_data_pipeline.deploy_results(state_results)
        dod_data_pipeline.deploy_results(county_results)

        _logger.info(f"Generated counties shape files for {intervention_enum.name}")

    print("finished dod job")
