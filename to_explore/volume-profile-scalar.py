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
def volume_profile_numba(prices, volumes, bins=100):
    # Calculate min and max prices
    price_min, price_max = np.min(prices), np.max(prices)
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, bins)
    
    # Initialize volume sum array
    volume_sum = np.zeros(len(price_bins) - 1)
    
    # Bin volumes into price ranges
    for i in range(len(prices)):
        for j in range(len(price_bins) - 1):
            if price_bins[j] <= prices[i] < price_bins[j + 1]:
                volume_sum[j] += volumes[i]
                break
    
    # Total volume and value area volume (70%)
    total_volume = np.sum(volume_sum)
    value_area_volume = 0.7 * total_volume
    
    # Sort by volume to determine Value Area
    sorted_indices = np.argsort(volume_sum)[::-1]
    cum_volume = np.cumsum(volume_sum[sorted_indices])
    
    # Determine Value Area High (VAH) and Low (VAL)
    vah_idx = np.argmax(cum_volume >= value_area_volume)
    vah = price_bins[sorted_indices[:vah_idx + 1]].max()
    val = price_bins[sorted_indices[:vah_idx + 1]].min()
    
    # Point of Control (POC) - Highest volume node
    poc_idx = np.argmax(volume_sum)
    poc = price_bins[poc_idx]
    
    return vah, val, poc


def get_volume_profile(data, days=7, bins=2000):
    # Extract last 'days' worth of data
    recent_data = data[-days*24:]  
    prices = recent_data['Close'].values
    volumes = recent_data['Volume'].values
    
    vah, val, poc = volume_profile_numba(prices, volumes, bins)
    
    return {'VAH': vah, 'VAL': val, 'POC': poc}
    

profile_levels = get_volume_profile(data, days=7)
print(f"VAH: {profile_levels['VAH']}, VAL: {profile_levels['VAL']}, POC: {profile_levels['POC']}")  

