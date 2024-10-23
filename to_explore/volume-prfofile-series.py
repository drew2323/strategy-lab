symbol='SPY'
start_date='one year ago'
timeframe='1h'
tz='America/New_York'
limit=50000

data = vbt.PolygonData.pull(
    symbol,
    start=start_date,
    timeframe=timeframe,
    tz=tz,
    limit=limit,
    missing_index="drop"    
).dropna()
data = data.get()
close = data['Close']
high = data['High']
low = data['Low']
open = data['Open']
vwap = data['VWAP']
volume = data['Volume']

@njit
def calculate_volume_profile(high_window, low_window, volume_window, num_bins):
    window_low = np.min(low_window)
    window_high = np.max(high_window)
    if window_low == window_high:
        window_low -= 0.0001  
        window_high += 0.0001
    bins = np.linspace(window_low, window_high, num_bins + 1)
    volume_profile = np.zeros(num_bins)
    for i in range(len(high_window)):
        bar_low = low_window[i]
        bar_high = high_window[i]
        bar_volume = volume_window[i]
        if bar_low == bar_high:
            bar_low -= 0.0001
            bar_high += 0.0001
        low_idx = np.searchsorted(bins, bar_low, side='left') - 1
        high_idx = np.searchsorted(bins, bar_high, side='right') - 1
        if high_idx < low_idx:
            high_idx = low_idx
        indices = np.arange(low_idx, high_idx + 1)
        if len(indices) > 0:
            volume_per_bin = bar_volume / len(indices)
            for idx in indices:
                if 0 <= idx < num_bins:
                    volume_profile[idx] += volume_per_bin
    return bins[:-1], volume_profile

@njit
def compute_value_area(bins, volume_profile):
    total_volume = np.sum(volume_profile)
    if total_volume == 0:
        return np.nan, np.nan, np.nan
    poc_idx = np.argmax(volume_profile)
    poc = bins[poc_idx]
    sorted_indices = np.argsort(volume_profile)[::-1]
    cumulative_volume = np.cumsum(volume_profile[sorted_indices])
    value_area_threshold = 0.7 * total_volume
    idx = np.searchsorted(cumulative_volume, value_area_threshold)
    value_area_indices = sorted_indices[:idx + 1]
    val = np.min(bins[value_area_indices])
    vah = np.max(bins[value_area_indices])
    return val, vah, poc

def calculate_val_vah_poc(high, low, volume, window_size=168, num_bins=1000):
    n = len(high)
    vals = np.full(n, np.nan)
    vahs = np.full(n, np.nan)
    pocs = np.full(n, np.nan)

    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1  
        high_window = high[start_idx:end_idx]
        low_window = low[start_idx:end_idx]
        volume_window = volume[start_idx:end_idx]
        bins, volume_profile = calculate_volume_profile(
            high_window, low_window, volume_window, num_bins
        )
        val, vah, poc = compute_value_area(bins, volume_profile)
        vals[i] = val
        vahs[i] = vah
        pocs[i] = poc

    return vals, vahs, pocs
  
  vals, vahs, pocs = calculate_val_vah_poc(high.values, low.values, volume.values, window_size=240, num_bins=2000)
