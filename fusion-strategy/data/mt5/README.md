# MT5 M1 export for the ORB test

Put the CSV files exported from your MetaTrader 5 **demo** terminal here, then push them to this branch.

## How to export (free, any MT5 broker)

1. In MT5 open **View → Symbols** (Ctrl+U), then the **Bars** tab.
2. Pick the symbol (XAUUSD may be called GOLD at your broker; suffixes like `.m` are fine), timeframe **M1**, and dates **2023-01-01 → today**. Click **Request**.
   - If your broker has less M1 history than that, note the first date shown. The test runs on whatever is there.
   - If fewer bars appear than expected, try setting **Tools → Options → Charts → Max bars in chart** to Unlimited and requesting again. I haven't verified that this setting affects the Bars tab.
3. Click **Export Bars** and save as `SYMBOL_YEAR.csv`. Use one file per symbol per year (for example `EURUSD_2023.csv`) so each file stays under GitHub's 100 MB limit. If the dialog only exports everything at once, one file per symbol is fine as long as it is under 100 MB.
4. Repeat for XAUUSD, EURUSD, GBPUSD and USDJPY.

The files must contain the `<SPREAD>` column, which the default export includes. Leave the format as MT5 writes it (UTF-16, tab-separated is fine).

## Also tell me (if you know)

- Your broker's **server timezone**, shown as the time in Market Watch versus your local time. Most brokers use GMT+2 in winter and GMT+3 in summer. The script detects this from the FX weekly open and prints what it found.
- Whether the account's commission matches the plan ($7 per round-turn lot). The test uses the plan's costs either way.

## Run it yourself (optional)

```
python -m venv .venv && .venv\\Scripts\\pip install pandas numpy pyarrow matplotlib
cd research
..\\.venv\\Scripts\\python orb_sessions.py --data ..\\data\\mt5 --out ..\\results\\orb_sessions
```
