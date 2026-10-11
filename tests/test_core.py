"""Tests for the LLM-free parts: tools + memory. Run: pytest -q"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from da_agent.data import STORE            # noqa: E402
from da_agent.memory import MemoryStore    # noqa: E402
import da_agent.tools as T                 # noqa: E402


def call(t, **kw):
    return t.invoke(kw)


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    STORE.load(ROOT / "sample_data" / "customers.csv")
    monkeypatch.setattr(T, "MEMORY", MemoryStore(tmp_path / "m.json"))
    monkeypatch.setattr(T, "OUTPUT_DIR", tmp_path)


def test_describe_and_query():
    assert "rows=600" in call(T.describe_dataset)
    assert "matching rows" in call(T.query_rows, filter_expr="plan == 'Pro'", limit=3)


def test_aggregate_matches_pandas():
    out = call(T.aggregate, metric_column="monthly_spend", agg="mean", group_by=["plan"])
    expected = STORE.df.groupby("plan")["monthly_spend"].mean().max()
    assert f"{expected:.4f}"[:6] in out


def test_trend_correlation_groups_outliers():
    assert "overall_change" in call(T.trend_over_time, date_column="signup_date", freq="Q")
    assert "satisfaction" in call(T.correlation_analysis, target="support_tickets")
    assert "ANOVA" in call(T.compare_groups, numeric_column="monthly_spend", group_column="plan")
    assert "outliers:" in call(T.detect_outliers, column="monthly_spend")


def test_plot_and_python(tmp_path):
    assert ".png" in call(T.plot_chart, kind="bar", x="plan", y="monthly_spend")
    assert "result = 600" in call(T.run_python, code="result = len(df)")
    with pytest.raises(Exception):
        call(T.run_python, code="import os")


def test_bad_column_gives_hint():
    with pytest.raises(Exception, match="Did you mean"):
        call(T.aggregate, metric_column="monthly_spen", agg="mean")


def test_memory_named_filter_is_applied_and_updated(tmp_path):
    m = T.MEMORY
    m.upsert("named_filter", "Active Customers", "status != 'cancelled'", STORE.name)
    n_active = int((STORE.df.status != "cancelled").sum())
    assert f"rows used: {n_active}" in call(T.aggregate, metric_column="*", agg="count", named_filter="active_customers")
    # conflict handling: redefinition updates and keeps history
    assert m.upsert("named_filter", "active_customers", "status == 'active'", STORE.name) == "updated"
    assert m.items[0]["history"][0]["value"] == "status != 'cancelled'"
    # persistence across "sessions"
    assert MemoryStore(m.path).named_filters(STORE.name)["active_customers"] == "status == 'active'"
    assert m.delete("active_customers") == 1


def test_works_on_second_dataset():
    STORE.load(ROOT / "sample_data" / "sales.csv")
    assert "overall_change" in call(T.trend_over_time, date_column="order_date", metric_column="revenue", agg="sum")
