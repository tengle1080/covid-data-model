import click
import logging
import os

from libs.pipelines import api_pipeline
from libs.datasets import dataset_filter
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention
from libs import us_state_abbrev

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-counties-api")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/output/counties",
    help="Output directory for artifacts",
)
@click.option(
    "--summary-output",
    default="results/output",
    help="Output directory for county summaries.",
)
def deploy_counties_api(disable_validation, input_dir, output, summary_output):
    """The entry function for invocation"""
    # check that the dirs exist before starting
    for directory in [input_dir, output, summary_output]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

    aggregate_level = AggregationLevel.COUNTY
    data_filter = dataset_filter.DatasetFilter(
        aggregate_level=aggregate_level,
        country="USA",
        states=list(us_state_abbrev.abbrev_us_state.keys())
    )
    filters = [data_filter]

    for intervention in list(Intervention):
        logger.info(f"Running intervention {intervention.name}")
        api_pipeline.run_for_intervention(aggregate_level, intervention, filters)


@click.command("county-fips-summaries")
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/county_summaries",
    help="Output directory for artifacts",
)
def county_fips_summaries(input_dir, output):
    """Generates sumary files by state and globally of counties with model output data."""
    county_summaries = api_pipeline.build_county_summary_from_model_output(input_dir)
    api_pipeline.deploy_results(county_summaries, output)
