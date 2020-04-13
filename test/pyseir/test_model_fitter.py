import os
import pandas as pd
from pyseir import OUTPUT_DIR
from pyseir.inference import model_fitter


def test_model_fitter():
    model_fitter.run_state(states_only=True, state='California')
    state_output_file = os.path.join(OUTPUT_DIR, 'pyseir', 'data', 'state_summary', f'summary_California_state_only__mle_fit_results.json')

    results = pd.read_json(state_output_file)
    assert results.iloc[0]['R0'] < 4
    assert results.iloc[0]['R0'] > .5
