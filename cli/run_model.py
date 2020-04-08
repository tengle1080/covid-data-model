#!/usr/bin/env python
import pathlib
import datetime
import logging
import click
from libs.datasets import data_version
import run

from libs.enums import Intervention
from libs.pipelines import dod_data_pipeline

_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command("county")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/county"),
)
@data_version.with_git_version_click_option
def run_county(
    version: data_version.DataVersion, output, state=None
):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    run.run_county_level_forecast(
        min_date, max_date, output, country="USA", state=state
    )
    if not state:
        version.write_file("county", output)
    else:
        _logger.info("Skip version file because this is not a full run")


@main.command("county-summary")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/county_summaries"),
)
@data_version.with_git_version_click_option
def run_county_summary(version: data_version.DataVersion, output, state=None):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    run.build_county_summary(min_date, output, state=state)

    # only write the version if we saved everything
    if not state:
        version.write_file("county_summary", output)
    else:
        _logger.info("Skip version file because this is not a full run")


@main.command("state")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/state"),
)
@data_version.with_git_version_click_option
def run_state(version: data_version.DataVersion, output, state=None):
    """Run State level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    run.run_state_level_forecast(min_date, max_date, output, country="USA", state=state)
    _logger.info(f"Wrote output to {output}")
    # only write the version if we saved everything
    if not state:
        version.write_file("states", output)
    else:
        _logger.info("Skip version file because this is not a full run")


@main.command('dod')
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
def run_dod_deploy(run_validation, input_file, output):
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python deploy_dod_dataset.py

    # deploy to the data bucket
    AWS_PROFILE=covidactnow BUCKET_NAME=data.covidactnow.org python deploy_dod_dataset.py

    # triggering persistance to local
    python deploy_dod_dataset.py
    """

    for intervention in list(Intervention):
        _logger.info(f"Starting to generate files for {intervention_enum.name}.")
        state_results, county_results = dod_data_pipeline.run_intervention(
            intervention, input_file, run_validation
        )
        dod_data_pipeline.deploy_results(state_results)
        dod_data_pipeline.deploy_results(county_results)

        _logger.info(f"Generated counties shape files for {intervention_enum.name}")

    print("finished dod job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
