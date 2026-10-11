"""Generates two small synthetic datasets so the agent can be shown on DIFFERENT data."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = Path(__file__).parent

# ---- customers: churn/spend analysis
n = 600
plan = rng.choice(["Free", "Basic", "Pro", "Enterprise"], n, p=[.35, .3, .25, .1])
base = {"Free": 0, "Basic": 20, "Pro": 60, "Enterprise": 200}
spend = np.array([base[p] for p in plan]) + rng.normal(0, 6, n)
spend = np.clip(spend, 0, None).round(2)
spend[rng.choice(n, 5, replace=False)] *= 8            # planted outliers
tickets = rng.poisson(2, n)
status = np.where(rng.random(n) < (0.10 + 0.04 * tickets), "cancelled",
                  np.where(rng.random(n) < 0.08, "paused", "active"))
cust = pd.DataFrame({
    "customer_id": range(1, n + 1),
    "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 640, n), unit="D"),
    "plan": plan, "region": rng.choice(["North", "South", "East", "West"], n),
    "status": status, "monthly_spend": spend, "support_tickets": tickets,
    "age": rng.integers(18, 70, n), "satisfaction": np.clip(10 - tickets + rng.normal(0, 1.5, n), 1, 10).round(1),
})
cust.loc[rng.choice(n, 25, replace=False), "satisfaction"] = np.nan   # missing values
cust.to_csv(out / "customers.csv", index=False)

# ---- sales: revenue/trend analysis (different columns on purpose)
m = 1200
prod = rng.choice(["Laptop", "Phone", "Tablet", "Headphones", "Monitor"], m)
price = {"Laptop": 900, "Phone": 600, "Tablet": 400, "Headphones": 120, "Monitor": 250}
units = rng.integers(1, 6, m)
disc = rng.choice([0, 0.05, 0.1, 0.2], m)
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, m), unit="D")
trend = 1 + (dates - dates.min()).days / 540 * 0.5    # sales grow over time
sales = pd.DataFrame({
    "order_date": dates, "product": prod, "category": np.where(np.isin(prod, ["Headphones", "Monitor"]), "Accessories", "Devices"),
    "region": rng.choice(["APAC", "EMEA", "AMER"], m, p=[.4, .25, .35]), "units": units, "discount": disc,
})
sales["revenue"] = (units * sales["product"].map(price) * (1 - disc) * trend).round(2)
sales.to_csv(out / "sales.csv", index=False)
print("wrote customers.csv and sales.csv")
