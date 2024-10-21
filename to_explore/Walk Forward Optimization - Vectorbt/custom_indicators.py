import numpy as np
import pandas as pd
import vectorbtpro as vbt
import talib
from numba import njit


# TREND FILTER INDICATOR
#   Apply Method
def tf_apply(close, ema_period, shift_period, smooth_period, tick_val):
    ma = vbt.nb.ma_1d_nb(close/tick_val, ema_period, wtype=2)
    rocma = ma - vbt.nb.fshift_1d_nb(ma, shift_period)
    tf = np.abs(vbt.nb.ma_1d_nb(rocma, smooth_period, wtype=2))
    return tf
#   Indicator Factor
TrendFilter = vbt.IF(
    class_name='TrendFilter',
    short_name='tf',
    input_names=['close'],
    param_names=['ema_period', 'shift_period', 'smooth_period', 'tick_val'],
    output_names=['real']
).with_apply_func(
    tf_apply,
    takes_1d=True,
    ema_period=200,
    shift_period=1,
    smooth_period=5,
    tick_val=1e-4
)

# Adaptive Laguerre
#   Helpers
@njit
def alag_loop_nb(p, l, warmup):
    f = np.full(p.shape, np.nan)
    L0 = np.full(p.shape, np.nan)
    L1 = np.full(p.shape, np.nan)
    L2 = np.full(p.shape, np.nan)
    L3 = np.full(p.shape, np.nan)
    dir_ = np.full(p.shape, 1)
    d = np.full(l, np.nan)

    for i in range(3, p.shape[0]):
        if i < warmup:
            f[i] = p[i]
            L0[i] = p[i]
            L1[i] = p[i-1]
            L2[i] = p[i-2]
            L3[i] = p[i-2]
        else:
            # Get Differences
            mi = 0
            mx = 0
            a = 0
            for j in range(l):
                d[j] = p[i-j] - f[i-j-1]
                mi = d[j] if d[j] < mi else mi
                mx = d[j] if d[j] > mx else mx
            # Min-Max Rescale
            d = (d - mi)/(mx - mi)
            a = np.nanmedian(d)

            # Calculation
            L0[i] = a*p[i] + (1-a)*L0[i-1]
            L1[i] = -(1 - a) * L0[i] + L0[i-1] + (1 - a) * L1[i-1]
            L2[i] = -(1 - a) * L1[i] + L1[i-1] + (1 - a) * L2[i-1]
            L3[i] = -(1 - a) * L2[i] + L2[i-1] + (1 - a) * L3[i-1]
            f[i] = (L0[i] + 2*L1[i] + 2*L2[i] + L3[i])/6
            if f[i] < f[i-1]:
                dir_[i] = -1

    return f, dir_

def alag_apply(high, low, timeperiod):
    # Hardcoded 3X Timeperiod Bar Warmup
    warm = 3*timeperiod

    # Average Price
    av_price = talib.MEDPRICE(high, low)
    av_price = vbt.nb.ffill_1d_nb(av_price)

    # Filter
    filter, direction = alag_loop_nb(av_price, timeperiod, warm)

    # Warmup
    filter[:warm] = np.nan
    direction[:warm] = 0

    # Trade Indicators
    conf = direction > 0
    long = vbt.nb.crossed_above_1d_nb(direction, np.full(high.shape, 0))
    short = vbt.nb.crossed_below_1d_nb(direction, np.full(high.shape, 0))

    return filter, direction, long, short, conf

AdaptiveLaguerre = vbt.IF(
    class_name='AdaptiveLaguerre',
    short_name='alag',
    input_names=['high', 'low'],
    param_names=['timeperiod'],
    output_names=['filter', 'direction', 'long', 'short', 'conf']
).with_apply_func(
    alag_apply,
    takes_1d=True,
    timeperiod=20,
)

class AdaptiveLaguerre(AdaptiveLaguerre):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(line_color='limegreen', connectgaps=False)),
             s_kwargs=dict(trace_kwargs=dict(line_color='red', connectgaps=False)),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}
        
        filter = self.select_col_from_obj(self.filter, column)
        direction = self.select_col_from_obj(self.direction, column)

        # Coloring
        long = pd.Series(np.nan, index=filter.index)
        short = pd.Series(np.nan, index=filter.index)
        ups = np.where(direction > 0)[0]
        downs = np.where(direction < 0)[0]

        # Back Shifting the Start of Each Sequence (proper coloring)
        # ups
        upstarts = ups[1:] - ups[:-1]
        upstarts = np.insert(upstarts, 0, upstarts[0]) == 1
        upstarts = ups[np.where(~upstarts & np.roll(upstarts,-1))[0]] - 1
        ups = np.append(ups, upstarts)

        # downs
        downstarts = downs[1:] - downs[:-1]
        downstarts = np.insert(downstarts, 0, downstarts[0]) == 1
        downstarts = downs[np.where(~downstarts & np.roll(downstarts,-1))[0]] - 1
        downs = np.append(downs, downstarts)

        # Plot Lines
        long[ups] = filter.iloc[ups]
        short[downs] = filter.iloc[downs]

        long.vbt.plot(fig=fig, **l_kwargs)
        short.vbt.plot(fig=fig, **s_kwargs)
        
        return fig
    

# AROON UP & DOWN
#   Helpers
def aroon_apply(h, l, t):

    u = 100*vbt.nb.rolling_argmax_1d_nb(h, t, local=True)/(t-1)
    d = 100*vbt.nb.rolling_argmin_1d_nb(l, t, local=True)/(t-1)

    u[:t] = np.nan
    d[:t] = np.nan

    # Trade Indicators
    conf = u > d
    long = vbt.nb.crossed_above_1d_nb(u, d)
    short = vbt.nb.crossed_below_1d_nb(u, d)

    return u, d, long, short, conf

Aroon = vbt.IF(
    class_name='Aroon',
    short_name='aroon',
    input_names=['high', 'low'],
    param_names=['timeperiod'],
    output_names=['up', 'down', 'long', 'short', 'conf']
).with_apply_func(
    aroon_apply,
    takes_1d=True,
    timeperiod=20,
)

# Class
class Aroon(Aroon):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(line_color='limegreen', connectgaps=False)),
             s_kwargs=dict(trace_kwargs=dict(line_color='red', connectgaps=False)),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}
        
        up = self.select_col_from_obj(self.up, column).rename('Aroon(up)')
        down = self.select_col_from_obj(self.down, column).rename('Aroon(down)')

        up.vbt.plot(fig=fig, **l_kwargs, **layout_kwargs)
        down.vbt.plot(fig=fig, **s_kwargs, **layout_kwargs)
        
        return fig
    
# Elhers Two Pole Super Smoother
#   Helpers
@njit
def tpss_loop(p, l, warmup):
    f = np.full(p.shape, np.nan)
    dir_ = np.full(p.shape, 1)

    # Initialize
    f[:warmup] = p[:warmup]
    a1 = np.exp(-1.414*3.14159/l)
    b1 = 2*a1*np.cos(np.deg2rad(1.414*180/l))
    c2 = b1
    c3 = -a1*a1
    c1 = 1 - c2 - c3
    for i in range(warmup, p.shape[0]):
        f[i] = c1*p[i] + c2*f[i-1] + c3*f[i-2]
        if f[i] - f[i-1] < 0:
            dir_[i] = -1

    return f, dir_

def tpss_apply(close, timeperiod):
    # Hardcoded 3X Timeperiod Bar Warmup
    warm = 3*timeperiod

    # Average Price
    av_price = (vbt.nb.fshift_1d_nb(close) + close)/2
    av_price = vbt.nb.ffill_1d_nb(av_price)

    # Filter
    filter, direction = tpss_loop(av_price, timeperiod, warm)

    # Warmup
    filter[:warm] = np.nan
    direction[:warm] = 0

    # Trade Indicators
    conf = direction > 0
    long = vbt.nb.crossed_above_1d_nb(direction, np.full(close.shape, 0))
    short = vbt.nb.crossed_below_1d_nb(direction, np.full(close.shape, 0))

    return filter, direction, long, short, conf

SuperSmoother = vbt.IF(
    class_name='SuperSmoother',
    short_name='tpss',
    input_names=['close'],
    param_names=['timeperiod'],
    output_names=['filter', 'direction', 'long', 'short', 'conf'],
    attr_settings=dict(
        filter=dict(dtype=np.float_),
        direction=dict(dtype=np.float_),
        long=dict(dtype=np.bool_),
        short=dict(dtype=np.bool_),
        conf=dict(dtype=np.bool_),
    )
).with_apply_func(
    tpss_apply,
    takes_1d=True,
    timeperiod=20
)

class SuperSmoother(SuperSmoother):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(line_color='limegreen', connectgaps=False)),
             s_kwargs=dict(trace_kwargs=dict(line_color='red', connectgaps=False)),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}
        
        filter = self.select_col_from_obj(self.filter, column)
        direction = self.select_col_from_obj(self.direction, column)

        # Coloring
        long = pd.Series(np.nan, index=filter.index)
        short = pd.Series(np.nan, index=filter.index)
        ups = np.where(direction > 0)[0]
        downs = np.where(direction < 0)[0]

        # Back Shifting the Start of Each Sequence (proper coloring)
        # ups
        upstarts = ups[1:] - ups[:-1]
        upstarts = np.insert(upstarts, 0, upstarts[0]) == 1
        upstarts = ups[np.where(~upstarts & np.roll(upstarts,-1))[0]] - 1
        ups = np.append(ups, upstarts)

        # downs
        downstarts = downs[1:] - downs[:-1]
        downstarts = np.insert(downstarts, 0, downstarts[0]) == 1
        downstarts = downs[np.where(~downstarts & np.roll(downstarts,-1))[0]] - 1
        downs = np.append(downs, downstarts)

        # Plot Lines
        long[ups] = filter.iloc[ups]
        short[downs] = filter.iloc[downs]

        long.vbt.plot(fig=fig, **l_kwargs)
        short.vbt.plot(fig=fig, **s_kwargs)
        
        return fig

# Kase Permission Stochastic
@njit
def kase_loop_nb(TripleK, pstX, alpha):
    TripleDF = np.full(TripleK.shape[0], 0.0)
    TripleDS = np.full(TripleK.shape[0], 0.0)

    for i in range(pstX+1, TripleK.shape[0]):
        TripleDF[i] = TripleDF[i-pstX] + alpha*(TripleK[i] - TripleDF[i-pstX])
        TripleDS[i] = (2.0*TripleDS[i-pstX] + TripleDF[i]) / 3.0
    return TripleDF, TripleDS

@njit
def kase_smooth_nb(price, length):
    # Initialization
    e0 = np.full(price.shape[0], 0.0)
    e1 = np.full(price.shape[0], 0.0)
    e2 = np.full(price.shape[0], 0.0)
    e3 = np.full(price.shape[0], 0.0)
    e4 = np.full(price.shape[0], 0.0)
    alpha = 0.45*(length-1.0)/(0.45*(length-1.0)+2.0)

    # Calculation
    for i in range(price.shape[0]):
        if i <= 2:
            e0[i] = price[i]
            e1[i] = price[i]
            e2[i] = price[i]
            e3[i] = price[i]
            e4[i] = price[i]
        else:
            e0[i] = price[i] + alpha * (e0[i-1] - price[i])
            e1[i] = (price[i] - e0[i]) * (1 - alpha) + alpha * e1[i-1]
            e2[i] = e0[i] + e1[i]
            e3[i] = e2[i] - e4[i-1] * (1-alpha)**2 + (alpha**2) * e3[i-1]
            e4[i] = e3[i] + e4[i-1]
        
    return e4

def kase_apply(h, l, c, pstLength, pstX, pstSmooth, smoothPeriod):

    # Variables
    lookback = pstLength*pstX
    alpha = 2.0/(1.0 + pstSmooth)

    # Calculations
    hh = vbt.nb.rolling_max_1d_nb(h, lookback)
    ll = vbt.nb.rolling_min_1d_nb(l, lookback)
    #  Triple K
    TripleK = vbt.nb.fillna_1d_nb(100*(c - ll)/(hh-ll), 0.0)
    TripleK[TripleK < 0] = 0.0

    #  Triple DF, DS
    TripleDF, TripleDS = kase_loop_nb(TripleK, pstX, alpha)

    #  SMA of DF and DS
    TripleDFs = talib.SMA(TripleDF, pstSmooth)
    TripleDSs = talib.SMA(TripleDS, pstSmooth)

    # Kase Smoothing
    pst = kase_smooth_nb(TripleDFs, smoothPeriod)
    pss = kase_smooth_nb(TripleDSs, smoothPeriod)

    # Signals and Confirmation
    long = vbt.nb.crossed_above_1d_nb(pst, pss)
    short = vbt.nb.crossed_above_1d_nb(pss, pst)
    conf = pst > pss

    return pst, pss, long, short, conf

KasePermissionStochastic = vbt.IF(
    class_name='KasePermissionStochastic',
    short_name='KPSS',
    input_names=['high', 'low', 'close'],
    param_names=['pstLength', 'pstX', 'pstSmooth', 'smoothPeriod'],
    output_names=['pst', 'pss', 'long', 'short', 'conf'],
    attr_settings=dict(
        pst=dict(dtype=np.float_),
        pss=dict(dtype=np.float_),
        long=dict(dtype=np.bool_),
        short=dict(dtype=np.bool_),
        conf=dict(dtype=np.bool_),
    )
).with_apply_func(
    kase_apply,
    takes_1d=True,
    pstLength=9,
    pstX=5,
    pstSmooth=3,
    smoothPeriod=10
)
class KasePermissionStochastic(KasePermissionStochastic):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             pst_kwargs=dict(trace_kwargs=dict(line_color='limegreen', connectgaps=False)),
             pss_kwargs=dict(trace_kwargs=dict(line_color='red', connectgaps=False)),
             fig=None,  
             **layout_kwargs):  
        pst_kwargs = pst_kwargs if pst_kwargs else {}
        pss_kwargs = pss_kwargs if pss_kwargs else {}

        pst = self.select_col_from_obj(self.pst, column).rename('KPSS-pst')
        pss = self.select_col_from_obj(self.pss, column).rename('KPSS-pss')
        long = self.select_col_from_obj(self.long, column).rename('Long')
        short = self.select_col_from_obj(self.short, column).rename('Short')

        pst.vbt.plot(fig=fig, **pst_kwargs, **layout_kwargs)
        pss.vbt.plot(fig=fig, **pss_kwargs, **layout_kwargs)
        long.vbt.signals.plot_as_entries(fig=fig, y=pst, **layout_kwargs)
        short.vbt.signals.plot_as_exits(fig=fig, y=pss, **layout_kwargs)
        
        return fig


## TREND LORD
def trend_lord_apply(c, period):

    # Variables
    sqrt_period = round(np.sqrt(period))

    # Calculations
    ma = vbt.nb.ma_1d_nb(c, period, wtype=vbt.enums.WType.Weighted)
    tl = vbt.nb.ma_1d_nb(ma, sqrt_period, wtype=vbt.enums.WType.Weighted)

    # Signals
    dir = np.sign(tl - vbt.nb.fshift_1d_nb(tl))
    long = vbt.nb.crossed_above_1d_nb(dir, np.full(c.shape[0], 0.0))
    short = vbt.nb.crossed_below_1d_nb(dir, np.full(c.shape[0], 0.0))
    conf = dir > 0

    return tl, dir, long, short, conf

TrendLord = vbt.IF(
    class_name='TrendLord',
    short_name='tl',
    input_names=['close'],
    param_names=['timeperiod'],
    output_names=['tl', 'direction', 'long', 'short', 'conf']
).with_apply_func(
    trend_lord_apply,
    takes_1d=True,
    timeperiod=20
)

class TrendLord(TrendLord):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(marker=dict(color='limegreen'))),
             s_kwargs=dict(trace_kwargs=dict(marker=dict(color='red'))),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}

        tl = self.select_col_from_obj(self.tl, column)
        direction = self.select_col_from_obj(self.direction, column)

        # Coloring
        long = pd.Series(np.nan, index=tl.index)
        short = pd.Series(np.nan, index=tl.index)
        ups = np.where(direction > 0)[0]
        downs = np.where(direction < 0)[0]

        # Plot Lines
        long[ups] = tl.iloc[ups]
        short[downs] = tl.iloc[downs]

        long.vbt.barplot(fig=fig, **l_kwargs, **layout_kwargs)
        short.vbt.barplot(fig=fig, **s_kwargs, **layout_kwargs)
        
        return fig

## Cyber Cycle
@njit
def cyber_cycle_apply(c, a):

    # Variables
    c = vbt.nb.ffill_1d_nb(c)
    smooth = np.full(c.shape[0], 0.0)
    cycle = np.full(c.shape[0], 0.0)
    trigger = np.full(c.shape[0], 0.0)

    # Calculations
    for i in range(0, c.shape[0]):
        if i < 4:
            cycle[i] = (c[i] - 2*c[i-1] + c[i-2])/4
            smooth[i] = (c[i] + 2*c[i-1] + 2*c[i-2] + c[i-3])/6
            trigger[i] = cycle[i-1]
        else:
            smooth[i] = (c[i] + 2*c[i-1] + 2*c[i-2] + c[i-3])/6
            cycle[i] = ((1 - 0.5*a)**2)*(smooth[i] - 2*smooth[i-1] + smooth[i-2]) + 2*(1-a)*cycle[i-1] - ((1-a)**2)*cycle[i-2]
            trigger[i] = cycle[i-1]

    # Remove Early Convergence Period
    cycle[:30] = 0.0
    trigger[:30] = 0.0

    # Signals
    long = vbt.nb.crossed_above_1d_nb(cycle, trigger)
    short = vbt.nb.crossed_below_1d_nb(cycle, trigger)
    conf = cycle > trigger

    return cycle, trigger, long, short, conf

CyberCycle = vbt.IF(
    class_name='CyberCycle',
    short_name='cc',
    input_names=['close'],
    param_names=['alpha'],
    output_names=['cycle', 'trigger', 'long', 'short', 'conf']
).with_apply_func(
    cyber_cycle_apply,
    takes_1d=True,
    alpha=0.7
)

class CyberCycle(CyberCycle):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(marker=dict(color='limegreen'))),
             s_kwargs=dict(trace_kwargs=dict(marker=dict(color='red'))),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}

        cycle = self.select_col_from_obj(self.cycle, column)
        trigger = self.select_col_from_obj(self.trigger, column)

        cycle.vbt.plot(fig=fig, **l_kwargs, **layout_kwargs)
        trigger.vbt.plot(fig=fig, **s_kwargs, **layout_kwargs)
        
        return fig

## Inverse Fisher of the Cyber Cycle
def icycle_apply(c, a, p, s, level=0.5):
    # Calculate Cyber Cycle
    cycle, _, _, _, _ = cyber_cycle_apply(c, a)

    # Scaling
    h = vbt.nb.rolling_max_1d_nb(cycle, p)
    l = vbt.nb.rolling_min_1d_nb(cycle, p)
    cycle = (2*5.0)*((cycle-l)/(h-l))-5.0

    # Smoothing
    cycle = talib.EMA(cycle, s)

    # Inverse Fisher of the Cyber Cycle
    icc = (np.exp(2*cycle)-1.0)/(np.exp(2*cycle)+1.0)

    # Signals
    long = vbt.nb.crossed_above_1d_nb(icc, np.full(cycle.shape[0], -1*level))
    short = vbt.nb.crossed_below_1d_nb(icc, np.full(cycle.shape[0], level))
    long_conf = icc < -1*level
    short_conf = icc > level
    
    return icc, long, short, long_conf, short_conf

iCyberCycle = vbt.IF(
    class_name='iCyberCycle',
    short_name='icc',
    input_names=['close'],
    param_names=['alpha', 'period', 'smoothing'],
    output_names=['icycle', 'long', 'short', 'long_conf', 'short_conf']
).with_apply_func(
    icycle_apply,
    takes_1d=True,
    alpha=0.7,
    period=10,
    smoothing=9
)

class iCyberCycle(iCyberCycle):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             c_kwargs=dict(trace_kwargs=dict(marker=dict(color='yellow'))),
             l_kwargs=dict(trace_kwargs=dict(marker=dict(color='limegreen'))),
             s_kwargs=dict(trace_kwargs=dict(marker=dict(color='red'))),
             fig=None,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}
        c_kwargs = c_kwargs if c_kwargs else {}

        icycle = self.select_col_from_obj(self.icycle, column)
        long_level = pd.DataFrame(np.full(len(icycle), -0.5), index=icycle.index)
        short_level = pd.DataFrame(np.full(len(icycle), 0.5), index=icycle.index)

        icycle.vbt.plot(fig=fig, **c_kwargs, **layout_kwargs)
        long_level.vbt.plot(fig=fig, **l_kwargs, **layout_kwargs)
        short_level.vbt.plot(fig=fig, **s_kwargs, **layout_kwargs)

        
        return fig

# Trend Trigger Factor
def t3_ma_apply(c, l, a=0.7):

    # Variables
    c1 = -a**3
    c2 = 3*a**2 + 3*a**3
    c3 = -6*a**2 - 3*a - 3*a**3
    c4 = 1 + 3*a + a**3 + 3*a**2

    # Calculation
    e1 = talib.EMA(c, l)
    e2 = talib.EMA(e1, l)
    e3 = talib.EMA(e2, l)
    e4 = talib.EMA(e3, l)
    e5 = talib.EMA(e4, l)
    e6 = talib.EMA(e5, l)
    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3

    return T3
def ttf_apply(c, l, t3=True, t3_period=3, b=0.7):

    # Fill Nan
    c = vbt.nb.ffill_1d_nb(c)

    # Variables
    c_s = vbt.nb.fshift_1d_nb(c, l)

    # Caluclations
    buyPower = vbt.nb.rolling_max_1d_nb(c, l) - vbt.nb.rolling_min_1d_nb(c_s, l)
    sellPower = vbt.nb.rolling_max_1d_nb(c_s, l) - vbt.nb.rolling_min_1d_nb(c, l)
    ttf = 100*((buyPower-sellPower)/(0.5*(buyPower+sellPower)))

    # T3 Smoothing
    if t3:
        ttf = t3_ma_apply(ttf, t3_period, a=b)

    # Signnals
    long = vbt.nb.crossed_above_1d_nb(ttf, np.full(c.shape[0], 0.0))
    short = vbt.nb.crossed_below_1d_nb(ttf, np.full(c.shape[0], 0.0))
    conf = ttf > 0

    return ttf, long, short, conf

TrendTriggerFactor = vbt.IF(
    class_name='TrendTriggerFactor',
    short_name='ttf',
    input_names=['close'],
    param_names=['timeperiod', 't3', 't3_period', 'b'],
    output_names=['ttf', 'long', 'short', 'conf']
).with_apply_func(
    ttf_apply,
    takes_1d=True,
    timeperiod=15,
    t3=True,
    t3_period=3,
    b=0.7
)

class TrendTriggerFactor(TrendTriggerFactor):
    from itertools import groupby, accumulate
    def plot(self, 
             column=None,  
             l_kwargs=dict(trace_kwargs=dict(marker=dict(color='limegreen'))),
             s_kwargs=dict(trace_kwargs=dict(marker=dict(color='red'))),
             n_kwargs=dict(trace_kwargs=dict(marker=dict(color='rgba(255, 255, 0, 0.1)'))),
             fig=None,
             signal=True,  
             **layout_kwargs):  
        l_kwargs = l_kwargs if l_kwargs else {}
        s_kwargs = s_kwargs if s_kwargs else {}
        n_kwargs = n_kwargs if n_kwargs else {}

        ttf = self.select_col_from_obj(self.ttf, column)

        if not signal:
            ttf.vbt.plot(fig=fig, **n_kwargs, **layout_kwargs)
        else:
            long = pd.DataFrame(np.full(len(ttf), 100), index=ttf.index)
            short = pd.DataFrame(np.full(len(ttf), -100), index=ttf.index)
            neutral = pd.DataFrame(np.full(len(ttf), np.nan), index=ttf.index)

            long_entry = self.select_col_from_obj(self.long, column).fillna(False)
            short_entry = self.select_col_from_obj(self.short, column).fillna(False)

            long_idx = long_entry
            short_idx = short_entry
            pre_long_idx = long_idx.shift(-1)
            pre_short_idx = short_idx.shift(-1)

            neutral[long_idx==True] = 100
            neutral[pre_long_idx==True] = -100
            neutral[short_idx==True] = -100
            neutral[pre_short_idx==True] = 100

            long[~self.select_col_from_obj(self.conf, column)] = np.nan
            short[self.select_col_from_obj(self.conf, column)] = np.nan

            long.vbt.plot(fig=fig, **l_kwargs, **layout_kwargs)
            short.vbt.plot(fig=fig, **s_kwargs, **layout_kwargs)
            neutral.vbt.plot(fig=fig, **n_kwargs, **layout_kwargs)

        
        return fig