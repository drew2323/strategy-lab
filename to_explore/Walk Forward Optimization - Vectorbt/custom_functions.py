import numpy as np
import pandas as pd
import vectorbtpro as vbt
import talib
from numba import njit
from pathlib import Path
import scipy
import itertools

FX_MAJOR_LIST = ['EURUSD','AUDNZD','AUDUSD','AUDJPY','EURCHF','EURGBP','EURJPY','GBPCHF','GBPJPY','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY','CADJPY','EURAUD','CHFJPY','EURCAD','AUDCAD','AUDCHF','CADCHF','EURNZD','GBPAUD','GBPCAD','GBPNZD','NZDCAD','NZDCHF','NZDJPY']
FX_MAJOR_LIST = sorted(FX_MAJOR_LIST)
FX_MAJOR_PATH = 'Data/FOREX/oanda/majors_{0}/{1}.csv'

# Major Currency Pair Loader
def get_fx_majors(datapath=FX_MAJOR_PATH, side='bid', start=None, end=None, fillna=True):
    if side.lower() not in ['bid', 'ask']:
        raise ValueError('Side *{0}* not recognized. Must be bid or ask'.format(side))
    print('Loading FOREX {0} major pairs.'.format(side.upper()))
    if '{0}' in datapath:
        data = vbt.CSVData.fetch([datapath.format(side.lower(), i) for i in FX_MAJOR_LIST], start=start, end=end)
    else:
        data = vbt.CSVData.fetch(['{0}/majors_{1}/{2}.csv'.format(datapath, side, i) for i in FX_MAJOR_LIST], start=start, end=end)
    
    return data

# FOREX Position Sizing with ATR
def get_fx_position_size(data, init_cash=10_000, risk=0.01, atr=14, sl=1.5):
    atr = vbt.talib("ATR").run(
            data.high,
            data.low,
            data.close,
            timeperiod=atr,
            skipna=True
        ).real.droplevel(0, axis=1)
    pip_decimal = {i:0.01 if 'JPY' in i else 0.0001 for i in data.close.columns}
    pip_value = {i:100 if 'JPY' in i else 10_000 for i in data.close.columns}
    stop_pips = np.ceil(sl*atr/pip_decimal)
    
    size = ((risk*init_cash/stop_pips)*pip_value)

    return size


# NNFX Dynamic Risk Model
@njit
def adjust_func_nb(c, atr, sl, tp):
    position_now = c.last_position[c.col]

    # Check if position is open and needs to be managed
    if position_now != 0:
        # Get Current SL & TP Info
        sl_info = c.last_sl_info[c.col]
        tp_info = c.last_tp_info[c.col]
        tp_info.ladder = True
        tsl_info = c.last_tsl_info[c.col]        

        last_order = c.order_records[c.order_counts[c.col] - 1, c.col]

        if last_order.stop_type == -1:
        # STOP TYPE == -1, User Generate Order
            # Get Current ATR Value
            catr = vbt.pf_nb.select_nb(c, atr)

            if not vbt.pf_nb.is_stop_info_active_nb(sl_info): 
                sl_info.stop = catr*vbt.pf_nb.select_nb(c, sl)

            if not vbt.pf_nb.is_stop_info_active_nb(tp_info):
                tp_info.stop = catr*vbt.pf_nb.select_nb(c, tp)
                tp_info.exit_size = round(abs(position_now) * 0.5)

        elif last_order.stop_type == vbt.sig_enums.StopType.TP:
        # STOP TYPE == 3, last fill was a take profit
            if not vbt.pf_nb.is_stop_info_active_nb(tsl_info):
                # Set a Trailing Stop for remaining
                tsl_info.stop = sl_info.stop

                # Deactivate Original Stop
                sl_info.stop = np.nan

def get_NNFX_risk(atr_, sl_, tp_):
    args = {"adjust_func_nb":adjust_func_nb,
            "adjust_args":(vbt.Rep("atr"), vbt.Rep("sl"), vbt.Rep("tp")),
            "broadcast_named_args":dict(
                atr=atr_,
                sl=sl_,
                tp=tp_
            ),
            "use_stops":True}
    
    return args


# Whites Reality Check for a Single Strategy
def col_p_val_WRC(col, means, n, inds):
    samples = col.values[inds].mean(axis=0)
    return (samples > means[col.name]).sum()/n

def get_WRC_p_val(raw_ret, allocations, n=2000):
    # Detrending & Zero Centering
    #raw_ret = np.log(data.close/data.open)
    det_ret = raw_ret - raw_ret.mean(axis=0)
    det_strat = np.sign(allocations+allocations.shift(1))*det_ret

    # Zero Centering
    mean_strat = det_strat.mean(axis=0)
    zero_strat = det_strat - mean_strat

    # Sampling
    inds = np.random.randint(0, raw_ret.shape[0], size=(raw_ret.shape[0], n))
    ps = zero_strat.apply(col_p_val_WRC, axis=0, args=(mean_strat, n, inds))

    return ps

# Monte Carlo Permutation Method (MCP) for Inference Testing
def col_p_val_MCP(col, det_ret, means, inds, n):
    samples = det_ret[col.name[-1]].values[inds]
    #print(col.values[:, np.newaxis].shape)
    samples = np.nanmean(samples*col.values[:, np.newaxis], axis=0)
    return (samples > means[col.name]).sum()/n

def get_MCP_p_val(raw_ret, allocations, n=2000):
    # Detrending
    det_ret = raw_ret - raw_ret.mean(axis=0)
    allocations = np.sign(allocations + allocations.shift(1))
    det_strat = allocations*det_ret

    # Zero Centering
    mean_strat = det_strat.mean(axis=0)

    # Sampling
    inds = np.tile(np.arange(0, raw_ret.shape[0])[:, np.newaxis], (1, 2000))
    inds = np.take_along_axis(inds, np.random.randn(*inds.shape).argsort(axis=0), axis=0)
    ps = allocations.apply(col_p_val_MCP, axis=0, args=(det_ret, mean_strat, inds, n))

    return ps

def _nonull_df_dict(df, times=True):
    if times:
        d = {i:df[i].dropna().to_numpy(dtype=int) for i in df.columns}
    else:
        d = {i:df[i].dropna().to_numpy() for i in df.columns}
    return d

def col_p_val_MCPH(col, det_ret, means, times, signals, n):
    # Get Column Specific Holding Times & Signals
    _times = times[col.name]
    _signals = signals[col.name]

    # Create Time/Signal Perumutation Arrays
    index_arr = np.tile(np.arange(0, len(_times))[:, np.newaxis], (1, n))
    sorter = np.random.randn(*index_arr.shape)
    index_arr = np.take_along_axis(index_arr, sorter.argsort(axis=0), axis=0)

    # Create Sampling Array
    _times_perm = _times[index_arr]
    _signals_perm = _signals[index_arr]
    _times_flat = _times_perm.flatten('F')
    _signals_flat = _signals_perm.flatten('F')
    samples = np.repeat(_signals_flat, _times_flat).reshape((len(col), n), order='F')
    samples = np.multiply(det_ret[col.name[-1]].values[np.tile(np.arange(0, col.shape[0])[:, np.newaxis], (1, n))], samples)

    samples = np.nanmean(samples, axis=0)

    return (samples > means[col.name]).sum()/n

def get_MCPH_p_val(raw_ret, allocations, n=2000):
    # Detrending
    #raw_ret = np.log(data.close/data.open)
    det_ret = raw_ret - raw_ret.mean(axis=0)
    allocations = np.sign(allocations + allocations.shift(1)).fillna(0)
    det_strat = allocations*det_ret

    # Zero Centering
    mean_strat = det_strat.mean(axis=0)

    # Strategy Allocation Holding Distribution and Corresponding Signals
    changes = (allocations == allocations.shift(1))
    times = changes.cumsum()-changes.cumsum().where(~changes).ffill().fillna(0).astype(int) + 1
    times = times[times - times.shift(-1, fill_value=1) >= 0]
    signals = allocations[~times.isnull()]

    # Get Dictionary of Times/Signals
    times = _nonull_df_dict(times)
    signals = _nonull_df_dict(signals, times=False)

    # Sampling
    ps = allocations.apply(col_p_val_MCPH, axis=0, args=(det_ret, mean_strat, times, signals, n))

    return ps

# Adjusted Returns: Adjusts closes of time series to reflect trade exit prices. Used as input to WRC and MCP statistical tests
def get_adjusted_returns(data, pf):
    # Trade Records
    records = pf.trades.records_readable[['Column', 'Exit Index', 'Avg Exit Price']]
    records.Column=records['Column'].apply(lambda x: x[-1])

    close_adj = data.get('Close')
    for row, value in records.iterrows():
        close_adj[value['Column']][value['Exit Index']] = value['Avg Exit Price']

    return np.log(close_adj/data.open)

# Optimized Split
def get_optimized_split(tf, frac, n):
    # Parameter Estimation
    d = tf/(frac + n*(1 - frac))
    di  = frac*d
    do = (1-frac)*d

    # Mixed Integer, Linear Optimization
    c = [-(1/frac - 1), 1]
    Aeq = [[1, n]]
    Aub = [[-1, 1],
           [(1/frac - 1), -1]]
    beq = [tf]
    bub = [0, 0]
    x0_bounds = (di*0.5, di*1.5)
    x1_bounds = (do*0.5, do*1.5)
    res = scipy.optimize.linprog(
        c, A_eq=Aeq, b_eq=beq, A_ub=Aub, b_ub=bub, bounds=(x0_bounds, x1_bounds),
        integrality=[1, 1],
        method='highs',
        options={"disp": True})

    # Solutions
    di, do = res.x

    # Actual Fraction
    frac_a = di/(do+di)

    return int(di), int(do), frac_a

def wfo_split_func(splits, bounds, index, length_IS=20, length_OOS=30):
    if len(splits) == 0:
        new_split = (slice(0, length_IS), slice(length_IS, length_OOS+length_IS))
    else:
        # Previous split, second set, right bound
        prev_end = bounds[-1][1][1]

        # Split Calculation
        new_split = (
            slice(prev_end-length_IS, prev_end),
            slice(prev_end, prev_end + length_OOS)
        )
    if new_split[-1].stop > len(index):
        return None
    return new_split

def get_wfo_splitter(index, fraction, n):
    # Generates a splitter based on train/(train+test) fraction and number of folds
    d_IS, d_OOS, frac = get_optimized_split(len(index), fraction, n)

    # Generate the Splitter
    splitter = vbt.Splitter.from_split_func(
            index,
            wfo_split_func,
            split_args=(
                vbt.Rep("splits"),
                vbt.Rep("bounds"),
                vbt.Rep("index"),
            ),
            split_kwargs={
                'length_IS':d_IS,
                'length_OOS':d_OOS
            },
            set_labels=["IS", "OOS"]
    )

    return splitter

# WFO Fold Analysis Splitters
def get_wfo_splitters(index, fractions, folds):
    # Create Combinations of Folds/Fractions
    combinations = itertools.product(fractions, folds)

    # Generate Splitters
    splitters = {}
    splitter_ranges = {}
    for comb in combinations:
        splitter = get_wfo_splitter(index, comb[0], comb[1])
        splitters.update({comb:splitter})
        splitter_ranges.update({comb:[d_IS, d_OOS, frac]})

    return splitters, splitter_ranges

# NNFX WFO Trainin Performance Function
@vbt.parameterized(merg_func='concat')
def strat_perf(data, ind, atr, pos_size, long_signal='long', short_signal='short', metric='sharpe_ratio'):
    # Simulation
    pf = vbt.Portfolio.from_signals(
        data,
        entries=getattr(ind, long_signal),
        short_entries=getattr(ind, short_signal),
        **get_NNFX_risk(atr, 1.5, 1.0),
        size=pos_size,
        size_type='amount',
        init_cash=10_000,
        delta_format='absolute',
        price='nextopen',
        stop_entry_price='fillprice',
        leverage=np.inf,
        #fixed_fees=pos_size*data.get('Spread')
    )  
    result = getattr(pf, metric)
    return result

# Walk Forward Optimization Portfolio Simulation
def walk_forward_optimization(data, ind, pos_size, atr, splitter, metric='total_return', long_signal='long', short_signal='short', group=True):
    
    # Calculate Performance on Training Sets
    train_perf = splitter.apply(
        strat_perf,
        vbt.Takeable(data),
        vbt.Takeable(ind),
        vbt.Takeable(atr),
        vbt.Takeable(pos_size),
        metric=metric,
        long_signal=long_signal,
        short_signal=short_signal,
        _execute_kwargs=dict(  
            show_progress=False,
            #clear_cache=50,  
            #collect_garbage=50
        ),
        merge_func='row_stack',
        set_='IS',
        execute_kwargs=dict(show_progress=True),
        jitted=True
    )

    # Get the Best Parameters
    exclusions = [i for i in range(len(train_perf.index.names)) if train_perf.index.names[i] not in getattr(ind, long_signal).columns.names]
    group = train_perf.groupby(['split','symbol'])
    best = group.idxmax()
    best[:] = [tuple([i[j] for j in range(len(i)) if j not in exclusions]) for i in best]
    best = best.droplevel('symbol')

    # Generate the OOS Signals
    opt_long = []
    opt_short = []
    for i in best.index.get_level_values('split').unique():
        _opt_long = splitter['OOS'].take(getattr(ind, long_signal))[i][best[i]]
        _opt_short = splitter['OOS'].take(getattr(ind, short_signal))[i][best[i]]

        remove_cols = [i for i in _opt_long.columns.names if i != 'symbol']
        _opt_long = _opt_long.droplevel(remove_cols, axis=1)
        _opt_short = _opt_short.droplevel(remove_cols, axis=1)

        opt_long.append(_opt_long)
        opt_short.append(_opt_short)
    opt_long = pd.concat(opt_long)
    opt_short = pd.concat(opt_short)

    # Run the WFO Portfolio
    group_by = len(opt_long.columns)*[0] if group else None
    pf = vbt.Portfolio.from_signals(
        data,
        entries=opt_long,
        short_entries=opt_short,
        **get_NNFX_risk(atr, 1.5, 1.0),
        size=pos_size,
        size_type='amount',
        init_cash=10_000,
        delta_format='absolute',
        price='nextopen',
        stop_entry_price='fillprice',
        leverage=np.inf,
        #fixed_fees=pos_size*data.get('Spread'),
        group_by=group_by
    )

    return pf

    # WFO Fold Analysis
def wfo_fold_analysis(data, ind, pos_size, atr, splitters, metric='total_return', long_signal='long', short_signal='short'):
    # Create the Results Matrix
    keys = splitters.keys()
    fractions = list(set([i[0] for i in keys]))
    folds = list(set([i[1] for i in keys]))
    FF, NN = np.meshgrid(fractions, folds)
    RR = np.zeros_like(FF)

    # Perform the Analysis
    for key, splitter in splitters.items():
        # Get the Key Indices
        idx = np.where((key[0] == FF) & (key[1] == NN))

        # WFO using Splitter
        print('Performing Walk Forward for train fraction {0} and N = {1}'.format(key[0], key[1]))
        wfo = walk_forward_optimization(data, ind, pos_size, atr, splitter, metric=metric, long_signal=long_signal, short_signal=short_signal)

        # Correlation
        rolling_returns = pd.DataFrame(wfo.cumulative_returns)
        rolling_returns = rolling_returns[rolling_returns != 1.0].dropna()
        rolling_returns['idx'] = np.arange(0, len(rolling_returns), 1)
        rolling_returns
        corr_matrix = rolling_returns.corr()
        R_sq = corr_matrix.iloc[0, 1]**2

        # Update the Results
        print(idx[0][0], idx[1][0], R_sq)
        RR[idx] = R_sq

    return FF, NN, RR