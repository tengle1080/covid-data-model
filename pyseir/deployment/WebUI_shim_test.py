import pytest
import numpy as np
from pyseir import load_data
from pyseir.inference.fit_results import load_inference_result
import pyseir.deployment.model_to_observed_shim as shim
import structlog


def test_connecticut(fips="06"):
    pyseir_outputs = load_data.load_ensemble_results(fips)
    fit_results = load_inference_result(fips)

    baseline_policy = "suppression_policy__inferred"
    model_acute_ts = pyseir_outputs[baseline_policy]["HGen"]["ci_50"]
    model_icu_ts = pyseir_outputs[baseline_policy]["HICU"]["ci_50"]
    idx_offset = int(fit_results["t_today"] - fit_results["t0"])
    observed_latest_dict = shim.get_latest_observed(fips)
    shim_log = structlog.getLogger(fips=fips)
    acute_shim, icu_shim = shim.shim_model_to_observations(
        model_acute_ts=model_acute_ts,
        model_icu_ts=model_icu_ts,
        idx=idx_offset,
        observed_latest=observed_latest_dict,
        log=shim_log,
    )
    assert not np.isnan(acute_shim), "Acute shim should not be nan"
    assert not np.isnan(icu_shim), "ICU shim should not be nan"


if __name__ == "__main__":
    test_connecticut()
