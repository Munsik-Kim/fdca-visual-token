"""Keep CPU roundoff portable without accepting changed evidence or identifiers."""
import csv
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'research/followup_2026_11'
EXPECTED = FOLLOWUP / 'evidence/lt16/expected/prediction_calibration.csv'
spec = importlib.util.spec_from_file_location('public_replay', FOLLOWUP / 'scripts/replay_public_evidence.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def altered(tmp_path, change):
    data = replay.rows(EXPECTED)
    change(data)
    path = tmp_path / 'calibration.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return path


def test_identical_calibration_is_counted_exactly():
    result = replay.compare_calibration(EXPECTED, EXPECTED)
    assert result['byte_identical']
    assert result['numeric_values'] == 8
    assert result['max_absolute_error'] == 0


def test_one_float_step_is_reported_and_accepted(tmp_path):
    def change(data):
        data[0]['weighted_MSE'] = math.nextafter(float(data[0]['weighted_MSE']), math.inf)
    result = replay.compare_calibration(altered(tmp_path, change), EXPECTED)
    assert not result['byte_identical']
    assert 0 < result['max_absolute_error'] < result['absolute_tolerance']


@pytest.mark.parametrize('error', ['changed_value', 'changed_identity', 'missing_row', 'nonfinite'])
def test_substantive_or_structural_changes_are_rejected(tmp_path, error):
    def change(data):
        if error == 'changed_value':
            data[0]['weighted_MSE'] = float(data[0]['weighted_MSE']) + 1e-8
        elif error == 'changed_identity':
            data[0]['predictor'] = 'different_predictor'
        elif error == 'missing_row':
            data.pop()
        else:
            data[0]['weighted_MSE'] = 'nan'
    with pytest.raises(AssertionError):
        replay.compare_calibration(altered(tmp_path, change), EXPECTED)
