import us
import structlog
from datetime import timedelta, datetime

from pyseir import load_data
from typing import Tuple

log = structlog.get_logger(__name__)

UNITY_SCALING_FACTOR = 1


def _get_model_to_dataset_conversion_factors_for_state(state, t0_simulation):
    """
    Return scaling factors to convert model hospitalization and model icu numbers to match
    the most current values provided in combined_datasets.

    Parameters
    ----------
    state
    t0_simulation
    Returns
    -------
    convert_model_to_observed_hospitalized
    convert_model_to_observed_icu
    """
    state_obj = us.states.lookup(state)

    # Get "Ground Truth" from outside datasets
    # NB: If only cumulatives are provided, we estimate current load. So this isn't strictly
    # actuals from covid-tracking.

    days_since_start, observed_latest_hospitalized = load_data.get_current_hospitalized_for_state(
        state=state_obj.abbr,
        t0=t0_simulation,
        category=load_data.HospitalizationCategory.HOSPITALIZED,
    )

    if observed_latest_hospitalized is None:
        # We have no observed data available. Best we can do is pass unity factors.
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    elif observed_latest_hospitalized == 0:
        # Right now our scaling factor can not capture this edge case
        log.msg(
            "Observed Hospitalized was 0 so we can not scale model outputs to latest observed",
            state=state,
        )
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    else:
        # Let's try to get a conversion for model to observed hospitalization

        # Rebuild date object
        t_latest_hosp_data_date = t0_simulation + timedelta(days=int(days_since_start))

        # Get Compartment Values for a Given Time
        model_state_hosp_gen = load_data.get_compartment_value_on_date(
            fips=state_obj.fips, compartment="HGen", date=t_latest_hosp_data_date
        )
        model_state_hosp_icu = load_data.get_compartment_value_on_date(
            fips=state_obj.fips, compartment="HICU", date=t_latest_hosp_data_date
        )

        # In the model, general hospital and icu hospital are disjoint states. We have to add them
        # together to get the correct comparable for hospitalized.
        model_heads_in_beds = model_state_hosp_gen + model_state_hosp_icu

        model_to_observed_hospitalized_ratio = observed_latest_hospitalized / model_heads_in_beds

        # Now let's look at ICU observed data
        _, observed_latest_icu = load_data.get_current_hospitalized_for_state(
            state=state_obj.abbr, t0=t0_simulation, category=load_data.HospitalizationCategory.ICU,
        )
        if observed_latest_icu is None:
            # We have observed hospitalizations, but not observed icu
            # We therefore scale ICU the same as general hospitalization
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        elif observed_latest_icu == 0:
            # Right now our scaling factor can not capture this edge case
            log.msg(
                "Observed ICU was 0. Falling back on Observed Hospitalization Ratio", state=state
            )
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        else:
            # We will have separate scaling factors. This is predicated on the assumption that we
            # should impose the location specific relative scaling factors instead of the model
            # derived ratio.
            model_to_observed_icu_ratio = observed_latest_icu / model_state_hosp_icu
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio


def _use_latest_get_model_to_dataset_conversion_factors_for_state(state, t0_simulation):
    """
    Return scaling factors to convert model hospitalization and model icu numbers to match
    the most current values provided in combined_datasets.

    Parameters
    ----------
    state
    t0_simulation
    Returns
    -------
    convert_model_to_observed_hospitalized
    convert_model_to_observed_icu
    """
    state_obj = us.states.lookup(state)

    # Get "Ground Truth" from outside datasets
    # NB: If only cumulatives are provided, we estimate current load. So this isn't strictly
    # actuals from covid-tracking.

    days_since_start, observed_latest_hospitalized = load_data.get_current_hospitalized_for_state(
        state=state_obj.abbr,
        t0=t0_simulation,
        category=load_data.HospitalizationCategory.HOSPITALIZED,
    )

    if observed_latest_hospitalized is None:
        # We have no observed data available. Best we can do is pass unity factors.
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    elif observed_latest_hospitalized == 0:
        # Right now our scaling factor can not capture this edge case
        log.msg(
            "Observed Hospitalized was 0 so we can not scale model outputs to latest observed",
            state=state,
        )
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    else:
        # Let's try to get a conversion for model to observed hospitalization

        # Rebuild date object
        t_latest_hosp_data_date = t0_simulation + timedelta(days=int(days_since_start))

        # Get Compartment Values for a Given Time
        model_state_hosp_gen = load_data.get_compartment_value_on_date(
            fips=state_obj.fips, compartment="HGen", date=t_latest_hosp_data_date
        )
        model_state_hosp_icu = load_data.get_compartment_value_on_date(
            fips=state_obj.fips, compartment="HICU", date=t_latest_hosp_data_date
        )

        # In the model, general hospital and icu hospital are disjoint states. We have to add them
        # together to get the correct comparable for hospitalized.
        model_heads_in_beds = model_state_hosp_gen + model_state_hosp_icu

        model_to_observed_hospitalized_ratio = observed_latest_hospitalized / model_heads_in_beds

        # Now let's look at ICU observed data
        _, observed_latest_icu = load_data.get_current_hospitalized_for_state(
            state=state_obj.abbr, t0=t0_simulation, category=load_data.HospitalizationCategory.ICU,
        )
        if observed_latest_icu is None:
            # We have observed hospitalizations, but not observed icu
            # We therefore scale ICU the same as general hospitalization
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        elif observed_latest_icu == 0:
            # Right now our scaling factor can not capture this edge case
            log.msg(
                "Observed ICU was 0. Falling back on Observed Hospitalization Ratio", state=state
            )
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        else:
            # We will have separate scaling factors. This is predicated on the assumption that we
            # should impose the location specific relative scaling factors instead of the model
            # derived ratio.
            model_to_observed_icu_ratio = observed_latest_icu / model_state_hosp_icu
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio


# def _get_model_to_dataset_conversion_factors_for_county(state, t0_simulation, fips, pyseir_outputs):
#     """
#     Return scaling factors to convert model hospitalization and model icu numbers to match
#     the most current values provided in combined_datasets.
#
#     Parameters
#     ----------
#     t0_simulation
#     fips
#     pyseir_outputs
#
#     Returns
#     -------
#     hosp_rescaling_factor
#     icu_rescaling_factor
#     """
#     # Check if we have county observed data
#     observed_county_hosp_latest, observed_county_icu_latest = _get_county_hospitalization(fips, t0_simulation)
#     log.info(
#         "Actual county hospitalizations",
#         fips=fips,
#         hospitalized=observed_county_hosp_latest,
#         icu=observed_county_icu_latest,
#     )
#     if observed_county_hosp_latest is None:
#         # What do we do if we have no county data? Fall back to the scales from the state?
#     elif observed_county_hosp_latest == 0:
#         # Can't scale. Do something else
#     else:
#         # If we have county hospitalization data, then we scale the county specific model's
#         # data to match
#         model_county_hosp_gen = load_data.get_compartment_value_on_date(
#             fips=fips,
#             compartment="HGen",
#             date=t_latest_hosp_data_date, #TODO: GET THIS
#             ensemble_results=pyseir_outputs,
#         )
#         model_county_icu = load_data.get_compartment_value_on_date(
#             fips=fips,
#             compartment="HICU",
#             date=t_latest_hosp_data_date,
#             ensemble_results=pyseir_outputs,
#         )
#         model_county_heads_in_beds = model_county_hosp_gen + model_county_icu
#
#         if observed_county_icu_latest is None:


#
#
#
#     state_abbreviation = us.states.lookup(state).abbr
#     t_latest_hosp_data, current_hosp_count = load_data.get_current_hospitalized_for_state(
#         state=state_abbreviation,
#         t0=t0_simulation,
#         category=load_data.HospitalizationCategory.HOSPITALIZED,
#     )
#
#     _, current_state_icu = load_data.get_current_hospitalized_for_state(
#         state=state_abbreviation,
#         t0=t0_simulation,
#         category=load_data.HospitalizationCategory.ICU,
#     )
#
#     if current_hosp_count is not None:
#         state_fips = fips[:2]
#         t_latest_hosp_data_date = t0_simulation + timedelta(days=int(t_latest_hosp_data))
#
#         state_hosp_gen = load_data.get_compartment_value_on_date(
#             fips=state_fips, compartment="HGen", date=t_latest_hosp_data_date
#         )
#         state_hosp_icu = load_data.get_compartment_value_on_date(
#             fips=state_fips, compartment="HICU", date=t_latest_hosp_data_date
#         )
#
#         if len(fips) == 5:
#             (observed_county_hosp_latest, observed_county_icu_latest,) = self._get_county_hospitalization(
#                 fips, t0_simulation
#             )
#             log.info(
#                 "Actual county hospitalizations",
#                 fips=fips,
#                 hospitalized=observed_county_hosp_latest,
#                 icu=observed_county_icu_latest,
#             )
#             inferred_county_hosp = load_data.get_compartment_value_on_date(
#                 fips=fips,
#                 compartment="HGen",
#                 date=t_latest_hosp_data_date,
#                 ensemble_results=pyseir_outputs,
#             )
#
#             county_hosp = inferred_county_hosp
#
#             inferred_county_icu = load_data.get_compartment_value_on_date(
#                 fips=fips,
#                 compartment="HICU",
#                 date=t_latest_hosp_data_date,
#                 ensemble_results=pyseir_outputs,
#             )
#             log.info(
#                 "Inferred county hospitalized for fips.",
#                 fips=fips,
#                 hospitalized=inferred_county_hosp,
#                 icu=inferred_county_icu,
#             )
#             county_icu = inferred_county_icu
#             if self._is_valid_count_metric(observed_county_hosp_latest):
#                 # use actual instead of adjusted
#                 county_hosp = observed_county_hosp_latest
#
#             if self._is_valid_count_metric(observed_county_icu_latest):
#                 county_icu = observed_county_icu_latest
#
#             # Rescale the county level hospitalizations by the expected
#             # ratio of county / state hospitalizations from simulations.
#             # We use ICU data if available too.
#             current_hosp_count *= (county_hosp + county_icu) / (state_hosp_gen + state_hosp_icu)
#
#         hosp_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)
#
#         # Some states have covidtracking issues. We shouldn't ground ICU cases
#         # to zero since so far these have all been bad reporting.
#         if len(fips) == 5 and self._is_valid_count_metric(observed_county_icu_latest):
#             icu_rescaling_factor = observed_county_icu_latest / inferred_county_icu
#         elif self._is_valid_count_metric(current_state_icu):
#             icu_rescaling_factor = current_state_icu / state_hosp_icu
#         else:
#             icu_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)
#     else:
#         hosp_rescaling_factor = 1.0
#         icu_rescaling_factor = 1.0
#     return hosp_rescaling_factor, icu_rescaling_factor
#
#     return NotImplementedError
#
#
# def _get_county_hospitalization(fips: str, t0_simulation: datetime) -> Tuple[float, float]:
#     """
#     Fetches the latest county hospitalization and icu utilization.
#
#     If current data is available, we return that.
#     If not, current values are estimated from cumulative.
#     """
#     county_hosp = load_data.get_current_hospitalized_for_county(
#         fips, t0_simulation, category=load_data.HospitalizationCategory.HOSPITALIZED,
#     )[1]
#     county_icu = load_data.get_current_hospitalized_for_county(
#         fips, t0_simulation, category=load_data.HospitalizationCategory.ICU
#     )[1]
#
#     return county_hosp, county_icu


if __name__ == "__main__":
    from pyseir.inference.fit_results import load_inference_result
    from libs.datasets import dataset_cache

    dataset_cache.set_pickle_cache_dir()
    # Need to have output artifacts from a build-all run

    # Run for California
    for state in us.STATES:
        fit_results = load_inference_result(state.fips)
        t0_simulation = datetime.fromisoformat(fit_results["t0_date"])

        a, b = _get_model_to_dataset_conversion_factors_for_state(state.name, t0_simulation)
        log.info(
            "Results for model to observed conversion:",
            state=state.name,
            hosp_rescaling_factor=round(a, 3),
            icu_rescaling_factor=round(b, 3),
        )

    # pyseir_outputs = load_data.load_ensemble_results()
    # Run those unittests
    # pass
