"""Helpers for notebooks/timestral_baseline.ipynb: zero-shot Timestral forecasts.

Metrics, baselines, interval calibration and plots are shared with lgbm_baseline.py.
"""

import numpy as np

from final.lgbm_baseline import DAY, H, WEATHER_COLUMNS

MODEL_ID = 'mistralai/Timestral-209M'
LEVEL = 0.95

MODEL_NAMES = {
    'timestral': 'Timestral + weather*',
    'timestral_uni': 'Timestral (no covariates)',
    'autoets': 'AutoETS',
    'naive': 'Seasonal naive',
}
MODEL_COLORS = {
    'timestral': '#d62728',
    'timestral_uni': '#ff9896',
    'autoets': '#9467bd',
    'naive': '#2ca02c',
}


def load_timestral(model_id=MODEL_ID, device=None):
    import torch
    from timestral import TimestralForecaster

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return TimestralForecaster.from_pretrained(model_id, device=device)


def to_numpy(x):
    """Forecast outputs are torch tensors on the model's device (GPU on Colab)."""
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def quantile_at(quantiles, levels, level):
    """Linear interpolation between the model's 21 quantile levels (0.025 is not one of them)."""
    return np.apply_along_axis(lambda row: np.interp(level, levels, row), -1, quantiles)


def timestral_forecast(model, data, origins, context_days, covariates=WEATHER_COLUMNS,
                       target_weather=None, level=LEVEL):
    """Zero-shot 48-hour forecasts for several origins in one batch.

    Context: y over the last context_days before each origin. Covariates are passed as
    known-future rows over context + horizon: past values from data, horizon values from
    target_weather[origin] (default: observed weather in data, i.e. optimistic).
    Returns {origin: (median, lower, upper)}, all clipped at 0.
    """
    origins = [int(o) for o in origins]
    target_weather = target_weather or {}
    y = data['y'].to_numpy(dtype=float)
    context = np.stack([y[o - context_days * DAY : o] for o in origins])[:, None, :]

    kwargs = {}
    if covariates:
        known = []
        for o in origins:
            future = target_weather.get(o, data.iloc[o : o + H])
            past = data.iloc[o - context_days * DAY : o]
            known.append(np.concatenate([past[covariates].to_numpy(float), future[covariates].to_numpy(float)]).T)
        kwargs['known_covariates'] = np.stack(known)

    forecast = model.forecast(context, prediction_length=H, **kwargs)
    quantiles = to_numpy(forecast.quantiles)[:, 0]  # (origins, H, 21)
    levels = to_numpy(forecast.quantile_levels).astype(float)
    median = to_numpy(forecast.median)[:, 0]
    lower = quantile_at(quantiles, levels, (1 - level) / 2)
    upper = quantile_at(quantiles, levels, (1 + level) / 2)
    return {
        o: (np.maximum(median[i], 0), np.maximum(lower[i], 0), np.maximum(upper[i], 0))
        for i, o in enumerate(origins)
    }


if __name__ == '__main__':
    levels = np.array([0.01, 0.05, 0.1])
    q = np.array([[1.0, 5.0, 10.0], [2.0, 6.0, 12.0]])
    assert np.allclose(quantile_at(q, levels, 0.025), [2.5, 3.5])
    assert np.allclose(quantile_at(q, levels, 0.05), [5.0, 6.0])
    print('ok')
