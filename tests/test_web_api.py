"""Unit tests for FastAPI Web Dashboard and REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from neuron_frontier.web.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_live_telemetry_endpoint():
    response = client.get("/api/telemetry/live")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "step" in data
    assert "val_bpb" in data
    assert "mfu_percent" in data


def test_judge_tour_endpoint():
    response = client.get("/api/judge_tour")
    assert response.status_code == 200
    data = response.json()
    assert len(data["steps"]) == 4
    assert "Hardware Co-Design" in data["steps"][0]["title"]


def test_autoresearch_pareto_endpoint():
    response = client.get("/api/autoresearch/pareto")
    assert response.status_code == 200
    data = response.json()
    assert "hypotheses" in data
    assert len(data["hypotheses"]) >= 1
