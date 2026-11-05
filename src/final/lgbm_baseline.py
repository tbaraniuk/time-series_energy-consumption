"""Helpers for notebooks/lightgbm_baseline.ipynb: features, baselines, metrics, plots."""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

H = 48
DAY = 24
WEATHER_COLUMNS = [
    'cloud_cover',
    'temperature',
    'wind_speed',
    'pressure',
    'humidity',
    'daylight',
    'air_density',
]
PARTS = {'1–24': slice(0, DAY), '25–48': slice(DAY, H), '1–48': slice(0, H)}
DAY_PARTS = ['1–24', '25–48']

MODEL_NAMES = {
    'weather': 'LightGBM + weather*',
    'autoets': 'AutoETS',
    'naive': 'Seasonal naive',
}
MODEL_COLORS = {
    'weather': '#1f77b4',
    'autoets': '#9467bd',
    'naive': '#2ca02c',
}
ACTUAL_COLOR = 'black'
HIDDEN_SPAN_COLOR = '#bbbbbb'


# Metrics

def rmse(actual, pred):
    errors = np.asarray(actual) - np.asarray(pred)
    return float(np.sqrt(np.mean(errors ** 2)))


def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def mape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def winkler(actual, lower, upper, alpha=0.05):
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    below = np.maximum(lower - actual, 0)
    above = np.maximum(actual - upper, 0)
    return float(np.mean(upper - lower + (2 / alpha) * (below + above)))


# Features and baselines

def make_features(y, origin, target_weather):
    """Build 48 rows using only target values observed before origin."""
    timestamp = pd.DatetimeIndex(target_weather.ds)
    h = np.arange(1, H + 1)
    within_day = (h - 1) % DAY
    previous_day = y[origin - DAY : origin]
    second_previous_day = y[origin - 2 * DAY : origin - DAY]
    previous_week_day = y[origin - 7 * DAY : origin - 6 * DAY]
    three_day_profile = y[origin - 3 * DAY : origin].reshape(3, DAY).mean(axis=0)
    recent = y[origin - 7 * DAY : origin]

    X = pd.DataFrame({
        'horizon': h,
        'hour': timestamp.hour.to_numpy(),
        'day_of_week': timestamp.day_of_week.to_numpy(),
        'day_of_year': timestamp.day_of_year.to_numpy(),
        'last_day_same_hour': previous_day[within_day],
        'two_days_ago_same_hour': second_previous_day[within_day],
        'week_ago_same_hour': previous_week_day[within_day],
        'last_3_days_same_hour': three_day_profile[within_day],
        'last_24h_mean': previous_day.mean(),
        'last_7d_mean': recent.mean(),
        'last_7d_std': recent.std(),
        'last_day_change': previous_day.mean() - second_previous_day.mean(),
    })
    for col in WEATHER_COLUMNS:
        X[col] = target_weather[col].to_numpy(dtype=float)
    return X


def seasonal_naive(y, origin):
    """Repeat the last observed day for both forecast days."""
    return np.tile(y[origin - DAY : origin], H // DAY)


def autoets(data, origin, train_days):
    """AutoETS(AAA, 24h) on log(y) over the last train_days, as in forecast.ipynb."""
    recent = data.iloc[origin - train_days * DAY : origin][['ds', 'y']].copy()
    recent['y'] = np.log(recent['y'])
    recent['unique_id'] = 'y'
    forecaster = StatsForecast(models=[AutoETS(season_length=DAY, model='AAA')], freq='h')
    return np.exp(forecaster.forecast(df=recent, h=H)['AutoETS'].to_numpy())


# Prediction intervals: 95th percentile of absolute CV errors per (forecast day, daylight flag)

def recent_level(y, origin, days=7):
    """Mean of y over the days before origin; used to rescale interval widths."""
    return float(np.mean(y[origin - days * DAY : origin]))


def calibrate_halfwidths(y, daylight, origins, predictions, level=0.95, level_days=None):
    """Return {(day_part, daylight_flag): half_width} from CV errors of one model.

    With level_days, errors are divided by recent_level at each origin, so the
    half-widths are fractions of the level and must be rescaled in interval().
    """
    widths = {}
    for part in DAY_PARTS:
        sl = PARTS[part]
        for flag in (0, 1):
            errors = np.concatenate([
                (np.abs(y[o : o + H][sl] - predictions[o][sl])
                 / (recent_level(y, o, level_days) if level_days else 1.0))[daylight[o : o + H][sl] == flag]
                for o in origins
            ])
            widths[(part, flag)] = float(np.quantile(errors, level, method='higher'))
    return widths


def interval(point, target_daylight, widths, scale=1.0):
    half = scale * np.array([
        widths[(DAY_PARTS[step // DAY], int(flag))]
        for step, flag in enumerate(target_daylight)
    ])
    return np.maximum(point - half, 0), point + half


# Plots

def plot_window(ax, ds, y, origin, forecasts, context_hours=DAY, band=None,
                names=MODEL_NAMES, colors=MODEL_COLORS):
    """Recent history, hidden actuals and forecasts for one 48-hour window.

    forecasts: {model_key: point}; band: optional (model_key, lower, upper).
    """
    hidden = ds.iloc[origin : origin + H]
    ax.axvspan(hidden.iloc[0], hidden.iloc[-1], color=HIDDEN_SPAN_COLOR, alpha=0.15, lw=0)
    ax.plot(ds.iloc[origin - context_hours : origin], y[origin - context_hours : origin],
            color='grey', lw=0.9, label='History')
    if band is not None:
        key, lower, upper = band
        ax.fill_between(hidden, lower, upper, color=colors[key], alpha=0.12, lw=0,
                        label=f'95% interval: {names[key]}')
    ax.plot(hidden, y[origin : origin + H], color=ACTUAL_COLOR, lw=2, label='Actual', zorder=5)
    for key, point in forecasts.items():
        ax.plot(hidden, point, color=colors[key], lw=1.4, label=names[key])
    ax.axvline(hidden.iloc[DAY], color='grey', ls=':', lw=1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


def plot_window_grid(fig, axes, ds, y, origins, forecasts_by_origin, titles, bands=None,
                     names=MODEL_NAMES, colors=MODEL_COLORS):
    """Small multiples of plot_window; the last free axis holds the legend."""
    flat = axes.ravel()
    for i, origin in enumerate(origins):
        band = bands[origin] if bands is not None else None
        plot_window(flat[i], ds, y, origin, forecasts_by_origin[origin], band=band,
                    names=names, colors=colors)
        flat[i].set_title(titles[i], fontsize=9)
    # Set after all panels are drawn so shared y-axes keep the tallest peak.
    flat[0].set_ylim(bottom=0)
    handles, labels = flat[0].get_legend_handles_labels()
    for ax in flat[len(origins):]:
        ax.axis('off')
    flat[-1].legend(handles, labels, loc='center', frameon=False)
