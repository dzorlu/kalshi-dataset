# FutureX Training Data from Kalshi Markets

This dataset prepares Kalshi prediction market data for training models to compete on the [FutureX benchmark](https://futurex-ai.github.io/).

## Methodology

**Goal**: Simulate the FutureX weekly submission format where predictions are made at a fixed weekly deadline for events resolving in the next 1-7 days.

### Data Pipeline

1. **Weekly Cutoffs**: Generate submission deadlines every Wednesday at 16:00 UTC (24:00 UTC+8)

2. **Event Selection**: For each cutoff, select Kalshi markets that:
   - Will resolve (expected_expiration_time) 1-7 days after the cutoff
   - Were already open at cutoff time
   - Meet quality criteria (see below)

3. **Quality Filters**:
   - **Uncertainty**: Price between 20-80¢ (genuine uncertainty, not obvious outcomes)
   - **Activity**: Volume > 1,000 contracts (real trading interest)
   - **Timing**: Didn't close >1 day before expected resolution (avoid early-certain markets)
   - **Valid outcomes**: Result is "yes" or "no"

4. **Training Examples**: Each example represents predicting a market outcome from the weekly cutoff point

### Dataset Statistics

- **Total examples**: 1,433
- **Unique weeks**: 46
- **Average per week**: 31.2 markets
- **Date range**: 2025 markets (Jan-Nov 2025)
- **Forecasting horizon**: 1-7 days (median: 4.2 days)
- **Outcome distribution**: 74.3% No, 25.7% Yes

### Features

Each training example includes:
- **Cutoff metadata**: cutoff_time, week_id
- **Market info**: ticker, title, category, rules
- **Market state**: price, volume, open_interest, liquidity, bid/ask spreads
- **Temporal**: days_until_resolution, days_since_open, market_duration
- **Label**: result (yes/no), settlement_value (0/100)

**Note**: `last_price` is used as proxy for price at cutoff. The source dataset doesn't include historical price snapshots, only final prices at market close. For most markets, this is a reasonable approximation since markets close near resolution time.

### Files

- `futurex_training_data.parquet` - Training data in parquet format
- `metadata.json` - Configuration and generation metadata
- `prepare_futurex_training_data.py` - Data generation script

## TODO

- [ ] **Event deduplication**: Some markets may represent the same underlying event with different strike prices (e.g., Bitcoin price ranges). Consider deduplicating or grouping related markets.
- [ ] Add historical price snapshots if available to get true price_at_cutoff
- [ ] Experiment with different uncertainty ranges (e.g., 15-85¢)
- [ ] Add market category features for better model contextualization

## Usage

```python
import pandas as pd

# Load training data
df = pd.read_parquet('data/futurex_training_data.parquet')

# Example: Train/test split by week
train_weeks = df['week_id'].unique()[:40]
train_df = df[df['week_id'].isin(train_weeks)]
test_df = df[~df['week_id'].isin(train_weeks)]
```

## Configuration

Current settings (see `metadata.json`):
- Cutoff day: Wednesday 16:00 UTC
- Forecast window: 1-7 days
- Uncertainty range: 20-80¢
- Min volume: 1,000 contracts
- Max early close: 1 day

To regenerate with different parameters, edit the constants in `prepare_futurex_training_data.py` and run:

```bash
python3 prepare_futurex_training_data.py
```
