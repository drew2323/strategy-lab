from numba import njit
import vectorbtpro as vbt
@njit
def signal_func_nb(c, entries, exits, short_entries, short_exits, cooldown_time, cooldown_bars):
    entry = vbt.pf_nb.select_nb(c, entries)
    exit = vbt.pf_nb.select_nb(c, exits)
    short_entry = vbt.pf_nb.select_nb(c, short_entries)
    short_exit = vbt.pf_nb.select_nb(c, short_exits)
    if not vbt.pf_nb.in_position_nb(c): #c.last_position == 0
        if vbt.pf_nb.has_orders_nb(c):  
            last_exit_idx = c.last_pos_info[c.col]["exit_idx"] #(92)  #If not in position, position information records contain information on the last (closed) position
            if cooldown_time is not None and c.index[c.i] - c.index[last_exit_idx] < cooldown_time:
                return False, exit, False, short_exit #disable entry
            elif cooldown_bars is not None and last_exit_idx + cooldown_bars > c.i:
                return False, exit, False, short_exit #disable entry
    return entry, exit, short_entry, short_exit