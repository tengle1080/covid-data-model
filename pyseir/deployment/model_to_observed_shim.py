# import structlog
from typing import Sequence
from datetime import datetime

import numpy as np

from pyseir import load_data
from pyseir.load_data import HospitalizationCategory


# log = structlog.getLogger()


def shim_model_to_observations(
    model_acute_ts: Sequence, model_icu_ts: Sequence, fips: str, t0: datetime, log
):
    """
    Take model outputs and shim them s.t. the latest observed value matches the same value for the
    outputted model.

    Parameters
    ----------
    model_acute_ts
        Model array sequence for acute prevalence
    model_icu_ts
        Model array sequence for icu prevalence
    fips
        Region of interest
    t0
        Datetime of beginning of model
    Return
    ------
    shimmed_acute: Sequence
        Transformed model_acute_ts Timeseries
    shimmed_icu: Sequence
        Transformed model_icu_ts Timeseries
    """
    f = _strict_match_model_to_observed
    # f = _intralevel_match_model_to_observed
    # f = _interlevel_match_model_to_observed
    return f(model_acute_ts, model_icu_ts, fips, t0, log=log)


# MOST SIMPLE OPTION
def _strict_match_model_to_observed(
    model_acute_ts: np.array, model_icu_ts: np.array, fips: str, t0: datetime, log
):
    """Most strict. Only shift if current value available at the correct aggregation level"""

    # Get Current Observed Value and Current Days Since Model Started
    ts_idx_acute, observed_latest_acute = load_data.get_current_hospitalized(
        fips=fips, t0=t0, category=HospitalizationCategory.HOSPITALIZED
    )  # Do we have to remove the icu from this? Assuming this is not including ICU but need to
    # confirm

    ts_idx_icu, observed_latest_icu = load_data.get_current_hospitalized(
        fips=fips, t0=t0, category=HospitalizationCategory.ICU
    )

    # Only Shim if Current Value is Provided
    if observed_latest_acute is None:
        acute_shim = 0
    else:
        acute_shim = observed_latest_acute - model_acute_ts[ts_idx_acute]

    if observed_latest_icu is None:
        icu_shim = 0
    else:
        icu_shim = observed_latest_icu - model_icu_ts[ts_idx_icu]

    shimmed_acute = acute_shim + model_acute_ts
    shimmed_clipped_acute = shimmed_acute.clip(min=0)
    shimmed_icu = icu_shim + model_icu_ts
    shimmed_clipped_icu = shimmed_icu.clip(min=0)

    log.info(
        "Model to Observed Shim Applied",
        fips=fips,
        acute_shim=np.round(acute_shim),
        icu_shim=np.round(icu_shim),
    )

    return shimmed_clipped_acute, shimmed_clipped_icu


def _interlevel_match_model_to_observed(
    model_acute_ts: np.array, model_icu_ts: np.array, fips: str, t0: datetime, log=None
):

    """
    Not yet implemented. Allow counties to draw from data from states. Needs to be clear about what
    to draw from. For example, if we don't have county ICU data, do we try to estimate it by using
    the state level icu_model to icu_observed ration? or county-level acute to icu ratio. There are
    many options
    """
    return NotImplementedError


#
# def match_model_to_observed(model_acute_ts: Sequence,
#                             model_icu_ts: Sequence,
#                             fips: str,
#                             t0: datetime,):
#     """
#
#     :param model_icu_ts:
#     :param model_acute_ts:
#     :param t0:
#     :param t0_simulation:
#     :param fips:
#     :return:
#     """
#     is_state = (len(fips) == 2)
#
#     # Get Current Observed Value and Current Days Since Model Started
#     days_since_start_acute, observed_latest_acute = load_data.get_current_hospitalized(
#         fips=fips,
#         t0=t0,
#         category=HospitalizationCategory.HOSPITALIZED
#     ) # Do we have to remove the icu from this? Assuming this is not including ICU but need to
#     # confirm
#
#     days_since_start_icu, observed_latest_icu = load_data.get_current_hospitalized(
#         fips=fips,
#         t0=t0,
#         category=HospitalizationCategory.ICU
#     )
#
#     if is_state:
#         # Simplest Case: Calculate Shift in Acute Cases
#         if observed_latest_acute is None:
#             state_acute_shift = 0
#         else:
#             #Vertical shift
#             state_acute_shift = observed_latest_acute - model_acute_ts[days_since_start_acute]
#
#         # Now check out icu
#         if observed_latest_icu is None:  # No Current ICU Available
#             if observed_latest_acute is None:  # Nor Current Acute Available
#                 state_icu_shift = 0
#             else: # No Current ICU but Current Acute Available
#                 # Convert State ICU using the ratio of model ICU to model acute to keep that ratio
#                 # the same
#                 numerator = model_icu_ts[days_since_start_acute]
#                 denominator = model_acute_ts[days_since_start_acute]
#                 convert_acute_to_icu_ratio =  numerator/ denominator
#                 state_icu_shift = convert_acute_to_icu_ratio * state_acute_shift
#         else: # ICU Data is Available
#             state_icu_shift = observed_latest_icu - model_acute_ts[days_since_start_icu]
#
#         shimmed_acute_ts = state_acute_shift + model_acute_ts
#         shimmed_icu_ts = state_icu_shift + model_icu_ts
#
#         return shimmed_acute_ts, shimmed_icu_ts
#
#     if not is_state:
#         if observed_latest_acute is None:
#             # Use state_acute_shift scaled from state to county
#         else:
#             county_acute_shift = observed_latest_acute - model_acute_ts[days_since_start_acute]
#
#         if observed_latest_icu is None:
#             # Use
#
#
#
#
#     else: is_county:
#
#     if observed_latest_value is None:
#         if is_state:
#             log.debug('No observed data available to shim model', fips=fips, category=category)
#             return x
#         else:
#             #find state value
#             days_since_start,  = _get_county_from_state_shim(fips=fips,
#                 t0=t0,
#                 category=category
#             )
#     else:
#
#
#
#     # Focus on states
#     if len(fips) == 2:
#
#     else:
#         return x
#
# def _get_county_from_state_shim(fips: str,
#                                 t0: datetime,
#                                 category: HospitalizationCategory):
#     """
#
#     :param fips:
#     :param t0:
#     :param category:
#     :return:
#     """
#
#     return
#
#
#
# def _get_model_to_dataset_conversion_factors(t0_simulation, fips, pyseir_outputs):
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
#     state_fips = fips[:2]
#
#     t_latest_hosp_data, current_hosp_count = load_data.get_current_hospitalized(
#         fips=state_fips,
#         t0=t0_simulation,
#         category=load_data.HospitalizationCategory.HOSPITALIZED,
#     )
#
#     _, current_state_icu = load_data.get_current_hospitalized(
#         fips=state_fips, t0=t0_simulation, category=load_data.HospitalizationCategory.ICU,
#     )
#
#     if current_hosp_count is not None:
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
#             (current_county_hosp, current_county_icu,) = self._get_county_hospitalization(
#                 fips, t0_simulation
#             )
#             log.info(
#                 "Actual county hospitalizations",
#                 fips=fips,
#                 hospitalized=current_county_hosp,
#                 icu=current_county_icu,
#             )
#             inferred_county_hosp = load_data.get_compartment_value_on_date(
#                 fips=fips,
#                 compartment="HGen",
#                 date=t_latest_hosp_data_date,  # this could be off by a day from the hosp data
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
#             if self._is_valid_count_metric(current_county_hosp):
#                 # use actual instead of adjusted
#                 county_hosp = current_county_hosp
#
#             if self._is_valid_count_metric(current_county_icu):
#                 county_icu = current_county_icu
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
#         if len(fips) == 5 and self._is_valid_count_metric(current_county_icu):
#             icu_rescaling_factor = current_county_icu / inferred_county_icu
#         elif self._is_valid_count_metric(current_state_icu):
#             icu_rescaling_factor = current_state_icu / state_hosp_icu
#         else:
#             icu_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)
#     else:
#         hosp_rescaling_factor = 1.0
#         icu_rescaling_factor = 1.0
#     return hosp_rescaling_factor, icu_rescaling_factor
#
#
#
#
#
#
#
#
#
# #### Brett's Other Scaling Code to use for names.
#
# UNITY_SCALING_FACTOR = 1
#
#
# def _get_model_to_dataset_conversion_factors_for_state(state, t0_simulation, fips):
#     """
#     Return scaling factors to convert model hospitalization and model icu numbers to match
#     the most current values provided in combined_datasets.
#     Parameters
#     ----------
#     state
#     t0_simulation
#     fips
#     Returns
#     -------
#     convert_model_to_observed_hospitalized
#     convert_model_to_observed_icu
#     """
#
#     # Get "Ground Truth" from outside datasets
#     # NB: If only cumulatives are provided, we estimate current load. So this isn't strictly
#     # actuals from covid-tracking.
#     state_abbreviation = us.states.lookup(state).abbr
#     days_since_start, observed_latest_hospitalized = load_data.get_current_hospitalized_for_state(
#         state=state_abbreviation,
#         t0=t0_simulation,
#         category=load_data.HospitalizationCategory.HOSPITALIZED,
#     )
#
#     if observed_latest_hospitalized is None:
#         # We have no observed data available. Best we can do is pass unity factors.
#         return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
#     elif observed_latest_hospitalized == 0:
#         # Right now our scaling factor can not capture this edge case
#         log.msg(
#             "Observed Hospitalized was 0 so we can not scale model outputs to latest observed",
#             state=state,
#         )
#         return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
#     else:
#         # Let's try to get a conversion for model to observed hospitalization
#
#         # Rebuild date object
#         t_latest_hosp_data_date = t0_simulation + timedelta(days=int(days_since_start))
#
#         # Get Compartment Values for a Given Time
#         model_state_hosp_gen = load_data.get_compartment_value_on_date(
#             fips=fips, compartment="HGen", date=t_latest_hosp_data_date
#         )
#         model_state_hosp_icu = load_data.get_compartment_value_on_date(
#             fips=fips, compartment="HICU", date=t_latest_hosp_data_date
#         )
#
#         # In the model, general hospital and icu hospital are disjoint states. We have to add them
#         # together to get the correct comparable for hospitalized.
#         model_heads_in_beds = model_state_hosp_gen + model_state_hosp_icu
#
#         model_to_observed_hospitalized_ratio = observed_latest_hospitalized / model_heads_in_beds
#
#         # Now let's look at ICU observed data
#         _, observed_latest_icu = load_data.get_current_hospitalized_for_state(
#             state=state_abbreviation,
#             t0=t0_simulation,
#             category=load_data.HospitalizationCategory.ICU,
#         )
#         if observed_latest_icu is None:
#             # We have observed hospitalizations, but not observed icu
#             # We therefore scale ICU the same as general hospitalization
#             model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
#             return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
#         elif observed_latest_icu == 0:
#             # Right now our scaling factor can not capture this edge case
#             log.msg(
#                 "Observed ICU was 0. Falling back on Observed Hospitalization Ratio", state=state
#             )
#             model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
#             return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
#         else:
#             # We will have separate scaling factors. This is predicated on the assumption that we
#             # should impose the location specific relative scaling factors instead of the model
#             # derived ratio.
#             model_to_observed_icu_ratio = observed_latest_icu / model_state_hosp_icu
#             return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
#
#
# def _get_model_to_dataset_conversion_factors_for_county(state, t0_simulation, fips):
#     """"""
#     return NotImplementedError
