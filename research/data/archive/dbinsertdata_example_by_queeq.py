"""Saves data from Kraken Futures, combining BTC and USD settled markets"""

import logging
from logging.config import fileConfig

from vectorbtpro import pd, vbt

from ext_lib.db import db_connect
from ext_lib.util import find_earliest_date

EXCHANGE = "1s_OHLCV"
SYMBOLS = ("BTC/USD:BTC", "BTC/USD:USD")
RESOLUTION = "1s"
DB_ENGINE = db_connect("ohlcv_1m")
DB_SYMBOL = "BTC/USD"

#fileConfig("logging.ini", disable_existing_loggers=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger("main")

vbt.CCXTData.set_custom_settings(exchange=EXCHANGE, timeframe=RESOLUTION, limit=6000000)
vbt.SQLData.set_engine_settings(engine_name="pg", engine=DB_ENGINE, schema=EXCHANGE, populate_=True, chunksize=1000)
vbt.SQLData.set_custom_settings(engine_name="pg", schema=EXCHANGE)


def main():
    if vbt.SQLData.has_table(DB_SYMBOL, schema=EXCHANGE):
        data = vbt.SQLData.has_table(DB_SYMBOL, schema=EXCHANGE)



    vbt.SQLData.create_schema(EXCHANGE)
    db_last_tstamp = None
    # TODO: figure out if it's possible to avoid using tables directly, but rather symbols
    if vbt.SQLData.has_table(DB_SYMBOL, schema=EXCHANGE):
        db_last_tstamp = vbt.SQLData.get_last_row_number(DB_SYMBOL, row_number_column="Open time")

    dfs = []
    for symbol in SYMBOLS:
        if db_last_tstamp is None:
            start = find_earliest_date(symbol, EXCHANGE)
        else:
            start = db_last_tstamp + pd.Timedelta(RESOLUTION)
        log.info("Start date for %s is %s", symbol, start)

        # Get data
        df = vbt.CCXTData.pull(symbol, exchange=EXCHANGE, timeframe=RESOLUTION, start=start).get()
        # If symbol is BTC/USD:BTC, convert volume from USD to BTC using the close price
        if symbol == "BTC/USD:BTC":
            df["Volume"] = df["Volume"].div(df["Close"])

        dfs.append(df)

    # Combine data from two symbols (last row is incomplete so removed)
    concatenated_data = pd.concat(dfs, axis=0)
    final_data = (
        concatenated_data.groupby(concatenated_data.index)
        .agg({"Open": "mean", "High": "mean", "Low": "mean", "Close": "mean", "Volume": "sum"})
        .iloc[:-1]
    )
    data = vbt.SQLData.from_data({DB_SYMBOL: final_data})
    # TODO: use custom method to prevent duplicate timestamps
    log.info("Saving to DB")
    vbt.SQLDataSaver(data).save_data(method="multi")


if __name__ == "__main__":
    main()