"""A fit without --device runs on the best available torch device."""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import cli_data


def _parse(*argv: str) -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	cli_data.add_args(parser)
	return parser.parse_args(["--input", "volume.lasagna.json", *argv])


@pytest.mark.parametrize(
	("cuda", "mps", "expected"),
	[(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
	monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
	monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
	assert cli_data.from_args(_parse()).device == expected


def test_explicit_device_is_kept(monkeypatch):
	monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
	monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
	assert cli_data.from_args(_parse("--device", "cpu")).device == "cpu"
	assert cli_data.from_args(_parse("--device", "cuda:1")).device == "cuda:1"
