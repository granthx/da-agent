"""Dataset loading + a tiny global store so tools stay dataset-agnostic."""
import re
from pathlib import Path

import pandas as pd

_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}/\d{1,2}/\d{2,4}")
_DATE_HINT = re.compile(r"(date|time|_at$|month|day)", re.I)


class DataStore:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.name: str = ""
        self.path: str = ""

    def load(self, path: str) -> pd.DataFrame:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        ext = p.suffix.lower()
        if ext in (".csv", ".txt"):
            df = pd.read_csv(p, sep=None, engine="python")  # sniff delimiter
        elif ext == ".tsv":
            df = pd.read_csv(p, sep="\t")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif ext == ".json":
            df = pd.read_json(p)
        elif ext == ".parquet":
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Use csv/tsv/xlsx/json/parquet.")
        df.columns = [str(c).strip() for c in df.columns]
        df = self._parse_dates(df)
        self.df, self.name, self.path = df, p.stem, str(p)
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if df[c].dtype != object:
                continue
            sample = df[c].dropna().astype(str).head(50)
            if sample.empty:
                continue
            looks_like_date = sample.str.match(_DATE_RE).mean() > 0.9
            if looks_like_date or (_DATE_HINT.search(c) and sample.str.len().between(6, 30).all()):
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[c] = parsed
        return df

    def schema_summary(self, df: pd.DataFrame | None = None, max_cols: int = 60) -> str:
        df = self.df if df is None else df
        if df is None:
            return "No dataset loaded."
        lines = [f"dataset='{self.name}' rows={len(df)} cols={df.shape[1]}"]
        for c in list(df.columns)[:max_cols]:
            s = df[c]
            ex = ", ".join(map(str, s.dropna().unique()[:4]))
            lines.append(f"- {c} [{s.dtype}] unique={s.nunique()} missing={int(s.isna().sum())} e.g. {ex}")
        if df.shape[1] > max_cols:
            lines.append(f"... {df.shape[1] - max_cols} more columns (use describe_dataset)")
        return "\n".join(lines)


STORE = DataStore()
