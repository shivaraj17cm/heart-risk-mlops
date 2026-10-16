"""DELIBERATE FAILURE - used only on branch `demo/ci-failure` to prove the CI gate.

This file must never be merged into main. It demonstrates that a failing unit test
turns the pipeline red and stops the downstream jobs (training and Docker build).
"""

from heartrisk.model_io import positive_proba  # real import: the test runs against project code


def test_deliberate_failure_demonstrates_ci_gate():
    assert callable(positive_proba)
    expected_classes = 3  # wrong on purpose: the target is binary (2 classes)
    assert expected_classes == 2, "Deliberate failure: demonstrates that CI stops on a failing test"
