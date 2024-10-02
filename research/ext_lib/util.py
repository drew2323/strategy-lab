import logging
from datetime import datetime, timedelta

from ccxt.base.errors import ExchangeNotAvailable
from vectorbtpro import pd, tp, vbt

log = logging.getLogger(__name__)


def find_earliest_date(symbol: str, exchange: str, **kwargs) -> tp.Optional[pd.Timestamp]:
    """Wrapper around CCXTData.find_earliest_date to handle ExchangeNotAvailable error with binary search
    Args:
        symbol: The trading symbol to query
        exchange: The exchange to query
        **kwargs: Additional arguments to pass to the find_earliest_date method

    Returns:
        tp.Optional[pd.Timestamp]: The earliest available date if found, otherwise None
    """
    log.info("Searching for earliest date for %s", symbol)
    start_date = pd.Timestamp(kwargs.pop("start", datetime(2010, 1, 1))).floor("D")
    end_date = pd.Timestamp(kwargs.pop("end", datetime.now())).floor("D")

    while start_date < end_date:
        log.info("Trying %s to %s range", start_date, end_date)
        mid_date = (start_date + (end_date - start_date) // 2).floor("D")
        try:
            found_date = vbt.CCXTData.find_earliest_date(
                symbol, exchange=exchange, start=mid_date, end=end_date, limit=10, **kwargs
            )
            if found_date:
                # Move the end date to mid_date to search the earlier half
                end_date = mid_date
            else:
                # Move the start date to mid_date + 1 to search the later half
                start_date = mid_date + timedelta(days=1)
        except ExchangeNotAvailable:
            # Move the start date to mid_date + 1 to search the later half
            start_date = mid_date + timedelta(days=1)

    # After the loop, start_date should be the earliest date with data
    try:
        found_date = vbt.CCXTData.find_earliest_date(
            symbol, exchange=exchange, start=start_date, end=end_date, **kwargs
        )
        return found_date
    except ExchangeNotAvailable as e:
        log.error("ExchangeNotAvailable error encountered at final step... Error: %s", e)
        return None
