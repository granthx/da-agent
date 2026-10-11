"""The agent's toolbox. Every tool computes on the real DataFrame and returns text.

Nine tools, seven different KINDS of analysis:
  describe_dataset     profiling / schema
  query_rows           filtering, sorting, row lookup
  aggregate            group-by statistics
  trend_over_time      time-series bucketing + growth
  correlation_analysis relationships between numeric columns
  compare_groups       statistical test: does a numeric metric differ across categories?
  detect_outliers      anomaly detection (IQR / z-score)
  plot_chart           visualisation (saves PNG)
  run_python           sandboxed pandas escape hatch (optional)

Tools raise ValueError with helpful messages; LangGraph's ToolNode feeds them back to the LLM
so it can fix its own arguments.
"""
import builtins
import contextlib
import difflib
import io
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from .config import ALLOW_PYTHON, MAX_TOOL_OUTPUT_CHARS, OUTPUT_DIR
from .data import STORE
from .memory import MEMORY, norm_key

# ------------------------------------------------------------------ helpers
def _df() -> pd.DataFrame:
    if STORE.df is None:
        raise ValueError("No dataset loaded.")
    return STORE.df


def _col(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        close = difflib.get_close_matches(str(name), list(df.columns), n=3)
        raise ValueError(f"Column '{name}' not found. Did you mean {close}? Columns: {list(df.columns)}")
    return name


def _subset(filter_expr: str = "", named_filter: str = ""):
    """Apply an optional saved named filter AND/OR an ad-hoc pandas query. Returns (df, description)."""
    df, exprs = _df(), []
    if named_filter:
        known = MEMORY.named_filters(STORE.name)
        key = norm_key(named_filter)
        if key not in known:
            raise ValueError(f"Unknown named_filter '{named_filter}'. Saved filters: {list(known) or 'none'}")
        exprs.append(known[key])
    if filter_expr:
        exprs.append(filter_expr)
    if not exprs:
        return df, "no filter"
    q = " and ".join(f"({e})" for e in exprs)
    try:
        return df.query(q, engine="python"), q
    except Exception as e:
        raise ValueError(f"Bad filter `{q}`: {e}. Use pandas .query() syntax; backtick column names with spaces.")


def _fmt(obj, max_rows: int = 30) -> str:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        obj = obj.round(4).to_string(max_rows=max_rows, max_colwidth=40)
    return str(obj)


def _out(text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        return text[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated; narrow the query]"
    return text


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    _col(df, col)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' is {df[col].dtype}, not numeric.")
    return df[col]


AGGS = {"count", "sum", "mean", "median", "min", "max", "std", "nunique"}

# ------------------------------------------------------------------ tools
@tool
def describe_dataset(columns: Optional[list[str]] = None) -> str:
    """Profile the dataset: shape, dtypes, missing values, unique counts and summary statistics.
    Use first when unsure about columns. columns: optional subset to describe."""
    df = _df()
    if columns:
        df = df[[_col(df, c) for c in columns]]
    stats = df.describe(include="all").T
    return _out(STORE.schema_summary(df) + "\n\nSUMMARY STATS:\n" + _fmt(stats, 60))


@tool
def query_rows(filter_expr: str = "", named_filter: str = "", columns: Optional[list[str]] = None,
               sort_by: Optional[str] = None, ascending: bool = True, limit: int = 10) -> str:
    """Filter/sort rows and show them, plus how many rows matched. Use for 'show me', 'which rows',
    'how many rows satisfy', or looking up specific records.
     Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    df, desc = _subset(filter_expr, named_filter)
    total = len(df)
    if sort_by:
        df = df.sort_values(_col(df, sort_by), ascending=ascending)
    if columns:
        df = df[[_col(df, c) for c in columns]]
    return _out(f"filter: {desc}\nmatching rows: {total} of {len(_df())}\n" + _fmt(df.head(min(limit, 50))))


@tool
def aggregate(metric_column: str, agg: str = "mean", group_by: Optional[list[str]] = None,
              filter_expr: str = "", named_filter: str = "", top_n: int = 20, sort_desc: bool = True) -> str:
    """Compute count/sum/mean/median/min/max/std/nunique of a column, optionally grouped by
    one or more columns. metric_column='*' with agg='count' counts rows. Use for totals, averages,
    'which X has the highest Y', percentages of a category (agg=count grouped, then divide).
    agg one of: count,sum,mean,median,min,max,std,nunique.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    if agg not in AGGS:
        raise ValueError(f"agg must be one of {sorted(AGGS)}")
    df, desc = _subset(filter_expr, named_filter)
    for g in group_by or []:
        _col(df, g)
    if metric_column == "*":
        if agg != "count":
            raise ValueError("metric_column='*' only works with agg='count'")
        res = df.groupby(group_by, dropna=False).size().rename("count") if group_by else len(df)
    else:
        _col(df, metric_column)
        if agg in {"sum", "mean", "median", "std"}:
            _numeric(df, metric_column)
        res = (df.groupby(group_by, dropna=False)[metric_column].agg(agg) if group_by
               else df[metric_column].agg(agg))
    if isinstance(res, pd.Series):
        res = res.sort_values(ascending=not sort_desc).head(top_n)
    return _out(f"filter: {desc}\nrows used: {len(df)}\n{agg}({metric_column})"
                f"{' by ' + ', '.join(group_by) if group_by else ''}:\n{_fmt(res)}")


@tool
def trend_over_time(date_column: str, metric_column: str = "*", agg: str = "sum", freq: str = "M",
                    filter_expr: str = "", named_filter: str = "") -> str:
    """Bucket a metric by time period and report growth. freq: D (day), W (week), M (month),
    Q (quarter), Y (year). Returns per-period values, % change, peak and overall change.
    metric_column='*' with agg='count' counts rows per period.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    df, desc = _subset(filter_expr, named_filter)
    _col(df, date_column)
    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError(f"'{date_column}' could not be parsed as dates.")
    if agg not in AGGS:
        raise ValueError(f"agg must be one of {sorted(AGGS)}")
    per = dates.dt.to_period(freq.upper())
    if metric_column == "*":
        series = df.groupby(per).size()
    else:
        _col(df, metric_column)
        series = df.groupby(per)[metric_column].agg(agg)
    t = pd.DataFrame({"value": series, "pct_change": series.pct_change() * 100})
    first, last = series.iloc[0], series.iloc[-1]
    change = f"{(last - first) / first * 100:.1f}%" if first else "n/a"
    summary = (f"periods={len(series)} first={first:.4g} last={last:.4g} overall_change={change} "
               f"peak={series.idxmax()} ({series.max():.4g}) trough={series.idxmin()} ({series.min():.4g})")
    return _out(f"filter: {desc}\n{agg}({metric_column}) per {freq.upper()}\n{summary}\n{_fmt(t, 40)}")


@tool
def correlation_analysis(columns: Optional[list[str]] = None, target: Optional[str] = None,
                         method: str = "pearson", filter_expr: str = "", named_filter: str = "") -> str:
    """Correlations between numeric columns. With target: rank all other numeric columns by
    correlation with it. Without: list the strongest pairs. method: pearson|spearman|kendall.
    Correlation is not causation.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    df, desc = _subset(filter_expr, named_filter)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman or kendall")
    num = df.select_dtypes("number")
    if columns:
        num = num[[_col(num, c) for c in columns]]
    if num.shape[1] < 2:
        raise ValueError("Need at least two numeric columns.")
    corr = num.corr(method=method)
    if target:
        _numeric(df, target)
        r = corr[target].drop(target).dropna()
        r = r.reindex(r.abs().sort_values(ascending=False).index)
        return _out(f"filter: {desc}\n{method} correlation with {target} (n={len(num)}):\n{_fmt(r.rename('r'))}")
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
             .rename("r").sort_values(key=abs, ascending=False).head(12))
    return _out(f"filter: {desc}\nstrongest {method} pairs (n={len(num)}):\n{_fmt(pairs)}")


@tool
def compare_groups(numeric_column: str, group_column: str, filter_expr: str = "",
                   named_filter: str = "") -> str:
    """Test whether a numeric column differs across categories of another column. Returns per-group
    count/mean/median/std, an ANOVA F-test p-value and eta-squared effect size. Use for
    'does X differ by Y', 'is there a significant difference between groups'.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    from scipy import stats
    df, desc = _subset(filter_expr, named_filter)
    _numeric(df, numeric_column)
    _col(df, group_column)
    g = df.groupby(group_column)[numeric_column]
    table = g.agg(["count", "mean", "median", "std"]).sort_values("count", ascending=False).head(20)
    groups = [x.dropna().values for _, x in g if x.notna().sum() >= 2]
    if len(groups) < 2:
        return _out(f"filter: {desc}\n{_fmt(table)}\nNot enough groups (>=2 with n>=2) for a test.")
    f, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    ss_b = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in groups)
    eta2 = ss_b / ((allv - allv.mean()) ** 2).sum()
    verdict = "significant at 5%" if p < 0.05 else "NOT significant at 5%"
    return _out(f"filter: {desc}\n{_fmt(table)}\nANOVA F={f:.3f} p={p:.4g} ({verdict}); eta_squared={eta2:.3f} "
                "(~0.01 small, 0.06 medium, 0.14 large)")


@tool
def detect_outliers(column: str, method: str = "iqr", threshold: Optional[float] = None,
                    filter_expr: str = "", named_filter: str = "") -> str:
    """Find anomalous values in a numeric column. method 'iqr' (default threshold 1.5) or 'zscore'
    (default threshold 3). Returns bounds, outlier count/percent and the most extreme rows.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    df, desc = _subset(filter_expr, named_filter)
    s = _numeric(df, column).dropna()
    if method == "iqr":
        k = threshold or 1.5
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (df[column] < lo) | (df[column] > hi)
    elif method == "zscore":
        k = threshold or 3.0
        mu, sd = s.mean(), s.std()
        lo, hi = mu - k * sd, mu + k * sd
        mask = ((df[column] - mu).abs() / sd) > k
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    out = df[mask.fillna(False)]
    worst = out.reindex((out[column] - s.median()).abs().sort_values(ascending=False).index).head(8)
    return _out(f"filter: {desc}\n{method} threshold={k}: normal range [{lo:.4g}, {hi:.4g}]\n"
                f"outliers: {len(out)} of {len(s)} ({len(out) / max(len(s), 1) * 100:.1f}%)\n"
                f"most extreme rows:\n{_fmt(worst.iloc[:, :8])}")


@tool
def plot_chart(kind: str, x: str, y: Optional[str] = None, agg: str = "mean", title: str = "",
               top_n: int = 15, filter_expr: str = "", named_filter: str = "") -> str:
    """Create a chart and save it as a PNG. kind: bar | line | hist | scatter | box.
    bar: x=category, y=numeric (aggregated with agg) or none for counts. line: x=date/ordered, y=numeric.
    hist: x=numeric. scatter: x,y both numeric. box: x=category, y=numeric.
    Returns the file path and the plotted numbers.  Filter args: filter_expr = optional pandas query such as plan == 'Pro' and age > 30; named_filter = name of a saved user-defined filter (see SAVED MEMORY)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, desc = _subset(filter_expr, named_filter)
    _col(df, x)
    fig, ax = plt.subplots(figsize=(8, 5))
    table = None
    if kind == "hist":
        _numeric(df, x).dropna().hist(ax=ax, bins=30)
    elif kind == "scatter":
        _numeric(df, x); _numeric(df, y or "")
        ax.scatter(df[x], df[y], s=12, alpha=0.6)
        ax.set_ylabel(y)
    elif kind == "box":
        _numeric(df, y or "")
        cats = df[x].value_counts().head(top_n).index
        data = [df.loc[df[x] == c, y].dropna() for c in cats]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels([str(c) for c in cats])
        ax.set_ylabel(y)
    elif kind in ("bar", "line"):
        key = df[x]
        if kind == "line" and pd.api.types.is_datetime64_any_dtype(key):
            key = key.dt.to_period("M").astype(str)
        if y:
            _numeric(df, y)
            table = df.groupby(key)[y].agg(agg)
        else:
            table = df.groupby(key).size()
        table = table.sort_values(ascending=False).head(top_n) if kind == "bar" else table.sort_index()
        table.plot(kind=kind, ax=ax)
        ax.set_ylabel(f"{agg}({y})" if y else "count")
    else:
        raise ValueError("kind must be bar, line, hist, scatter or box")
    ax.set_xlabel(x)
    ax.set_title(title or f"{kind}: {y + ' by ' if y else ''}{x}")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{kind}_{int(time.time() * 1000)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _out(f"Chart saved to {path} (filter: {desc})" + (f"\nPlotted values:\n{_fmt(table)}" if table is not None else ""))


_BLOCK = re.compile(r"(__|\bimport\b|\bopen\s*\(|\bexec\b|\beval\b|\bcompile\b|\bgetattr\b|\bsetattr\b|"
                    r"\bglobals\b|\blocals\b|\binput\b|\bbreakpoint\b|\b(os|sys|subprocess|shutil|socket)\b|"
                    r"\.(read_\w+|to_(csv|pickle|excel|parquet|json|sql|feather|hdf|clipboard)))")
_SAFE_BUILTINS = {n: getattr(builtins, n)
                  for n in ("abs all any bool dict enumerate float int isinstance len list map max min "
                            "print range round set sorted str sum tuple zip").split()}


@tool
def run_python(code: str) -> str:
    """LAST RESORT for analyses the other tools cannot do (custom ratios, multi-step logic).
    Write pandas code using `df` (the full DataFrame), `pd` and `np`. Assign the answer to a variable
    named `result`. No imports, no file/network access. Prefer the other tools when they fit."""
    if _BLOCK.search(code):
        raise ValueError("Code rejected by safety filter (imports, dunder, file/OS access are not allowed).")
    env = {"__builtins__": _SAFE_BUILTINS, "df": _df().copy(), "pd": pd, "np": np}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, env)  # noqa: S102  - filtered + restricted, but NOT a hardened sandbox
    parts = [buf.getvalue().strip()]
    if "result" in env:
        parts.append("result = " + _fmt(env["result"]))
    return _out("\n".join(p for p in parts if p) or "Code ran but produced no output; assign to `result`.")


def get_tools() -> list:
    tools = [describe_dataset, query_rows, aggregate, trend_over_time, correlation_analysis,
             compare_groups, detect_outliers, plot_chart]
    if ALLOW_PYTHON:
        tools.append(run_python)
    return tools
