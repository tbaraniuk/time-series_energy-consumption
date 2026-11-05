# Forecasting electricity generation for the next 48 hours

**MATH840 time series final project · Kyiv School of Economics**

## The problem

An energy company needs an hourly forecast of its electricity **generation** for the next two days. For each of the 48 hours, we provide a predicted value and a 95% prediction interval. The interval shows the range of plausible outcomes and helps the grid operator plan reserves.

The history ends at **2026-10-27 00:00**. The forecast covers **2026-10-27 01:00 through 2026-10-29 00:00**. The assignment calls the target `y` and does not specify a physical unit for it.

![Observed generation over five days, followed by the saved 48-hour forecast and its 95% prediction interval.](assets/forecast-overview.png)

The recent history shows a clear daily pattern: high daytime generation and much lower values at night.

## Data

| File | What it contains |
| --- | --- |
| [`data/train.csv`](data/train.csv) | 33,480 hourly observations from January 2023 to October 2026: time (`ds`), generation (`y`), and observed weather. |
| [`data/weather_forecast.csv`](data/weather_forecast.csv) | Forecast weather for the 48 hours we need to predict. It has no `y` values. |

The weather columns are cloud cover, temperature, wind speed, pressure, humidity, daylight, and air density. The current AutoETS blend uses past generation; the supplied future weather is available for other approaches.

## Our approach

1. **Seasonal naïve baseline:** repeat the generation from the same hour of the previous day.
2. **AutoETS:** fit an exponential smoothing model with a 24-hour season to the recent history.
3. **Blend:** compare a 50/50 average of the first two forecasts. The saved forecast uses a three-way average of seasonal naïve, AutoETS, and the mean of the same hour over the previous three days.
4. **Uncertainty:** start with the seasonal naïve interval, scale its width to 70% in the saved forecast, and keep the lower bound at zero or above.

We plan to try **LightGBM** as another model. It has not been implemented in this repository yet, so no LightGBM results are claimed here.

### Check on held-back data

We held back the last 48 hours of `data/train.csv` and forecast them using earlier data. On this **one validation window**, the 50/50 AutoETS + seasonal naïve blend had the lowest point-forecast error:

![RMSE on the last 48 held-back hours: seasonal naïve 120.4, AutoETS 121.9, and the 50/50 blend 114.2. Lower is better.](assets/validation-rmse.png)

| Model | RMSE ↓ | MAPE ↓ | 95% interval coverage | Winkler ↓ |
| --- | ---: | ---: | ---: | ---: |
| Seasonal naïve | 120.4 | 51.1% | 100% | 568.5 |
| AutoETS | 121.9 | 41.4% | 93.8% | 887.2 |
| 50/50 blend | **114.2** | **39.1%** | 100% | 568.5 |

These are validation results from the two-model comparison in the notebook. They are not scores on the hidden 48 hours or on the planned LightGBM model. Interval coverage on one window is only a limited check of a nominal 95% interval.

## Run the project

The working notebook is [`notebooks/forecast.ipynb`](notebooks/forecast.ipynb). It reads the two input CSV files from `data/`, explores the series, compares models, plots the forecast, and writes `output/BARANIUK_project.csv` with the required columns: `ds`, `forecast`, `lower_95`, and `upper_95`.

With [uv](https://docs.astral.sh/uv/) installed, run:

```bash
uv sync
uv run --with notebook jupyter notebook notebooks/forecast.ipynb
```

Run the notebook cells in order. The saved output is [`output/BARANIUK_project.csv`](output/BARANIUK_project.csv); the other `output/BARANIUK_project_*.csv` files are earlier forecast variants.

## LLM use

I used an LLM to help generate code for this project. This README was also drafted with LLM assistance and checked against the repository files.
