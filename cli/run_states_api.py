#!/usr/bin/env python
import click
import logging
import os

from libs.pipelines import api_pipeline
from libs.datasets import dataset_filter
from libs.datasets.dataset_utils import AggregationLevel
from libs import us_state_abbrev
from libs.enums import Intervention

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-states-api")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir", "-i", default="results", help="Input directory of state projections",
)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
)
@click.option(
    "--summary-output",
    default="results/output",
    help="Output directory for state summaries.",
)
def deploy_states_api(disable_validation, input_dir, output, summary_output):
    """The entry function for invocation"""

    # check that the dirs exist before starting
    for directory in [input_dir, output, summary_output]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

    aggregate_level = AggregationLevel.STATE
    data_filter = dataset_filter.DatasetFilter(
        aggregate_level=aggregate_level,
        country="USA",
        states=list(us_state_abbrev.abbrev_us_state.keys())
    )
    filters = [data_filter]

    for intervention in list(Intervention):
        logger.info(f"Running intervention {intervention.name}")
        api_pipeline.run_for_intervention(aggregate_level, intervention, filters)
