"""
FutureX-style Training Data Preparation for Kalshi Markets

This script prepares Kalshi prediction market data for training models
to compete on the FutureX benchmark. It simulates weekly submission deadlines
and creates training examples that predict market outcomes 1-7 days in advance.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict

# Configuration
CUTOFF_DAY = 'W-WED'  # Weekly on Wednesdays
CUTOFF_HOUR = 16  # 24:00 UTC+8 = 16:00 UTC
FORECAST_WINDOW_DAYS = (1, 7)  # Predict events 1-7 days ahead
UNCERTAINTY_RANGE = (20, 80)  # Only include markets with prices in this range
MIN_VOLUME = 1000  # Minimum trading volume threshold
MAX_EARLY_CLOSE_DAYS = 1  # Exclude markets that closed more than this many days early

def load_kalshi_data():
    """Load and preprocess Kalshi dataset"""
    print("Loading Kalshi dataset...")
    dataset = load_dataset('SubconsciousDev/predict')
    df = dataset['train'].to_pandas()

    # Convert time columns to datetime
    time_cols = ['open_time', 'close_time', 'expected_expiration_time', 'expiration_time']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format='ISO8601')

    # Calculate derived features
    df['market_duration_days'] = (df['close_time'] - df['open_time']).dt.total_seconds() / 86400
    df['closed_early_days'] = (df['expected_expiration_time'] - df['close_time']).dt.total_seconds() / 86400

    print(f"Loaded {len(df)} markets")
    return df

def filter_quality_markets(df):
    """Filter for high-quality, uncertain markets"""
    filtered = df[
        (df['last_price'] >= UNCERTAINTY_RANGE[0]) &
        (df['last_price'] <= UNCERTAINTY_RANGE[1]) &
        (df['volume'] > MIN_VOLUME) &
        (df['closed_early_days'] < MAX_EARLY_CLOSE_DAYS) &
        (df['result'].isin(['yes', 'no']))  # Valid outcomes only
    ].copy()

    print(f"After quality filtering: {len(filtered)} markets ({len(filtered)/len(df)*100:.1f}%)")
    return filtered

def generate_weekly_cutoffs(df):
    """Generate weekly cutoff timestamps"""
    min_date = df['open_time'].min()
    max_date = df['expected_expiration_time'].max()

    # Generate weekly cutoffs (Wednesdays at 16:00 UTC)
    cutoff_dates = pd.date_range(
        start=min_date.floor('D'),
        end=max_date,
        freq=CUTOFF_DAY
    )

    # Add specific time (16:00 UTC = 24:00 UTC+8)
    cutoffs = [dt + pd.Timedelta(hours=CUTOFF_HOUR) for dt in cutoff_dates]

    print(f"Generated {len(cutoffs)} weekly cutoffs from {cutoffs[0]} to {cutoffs[-1]}")
    return cutoffs

def create_training_examples(df, cutoffs):
    """Create training examples for each weekly cutoff"""
    training_data = []
    weekly_stats = defaultdict(lambda: {'total_markets': 0, 'yes_outcomes': 0, 'no_outcomes': 0})

    for cutoff in cutoffs:
        # Define forecast window
        window_start = cutoff + pd.Timedelta(days=FORECAST_WINDOW_DAYS[0])
        window_end = cutoff + pd.Timedelta(days=FORECAST_WINDOW_DAYS[1])

        # Find markets that:
        # 1. Will resolve within the forecast window
        # 2. Were already open at cutoff time
        # Note: We don't require market to still be trading - use final data if closed
        weekly_markets = df[
            (df['expected_expiration_time'] >= window_start) &
            (df['expected_expiration_time'] <= window_end) &
            (df['open_time'] <= cutoff)
        ].copy()

        if len(weekly_markets) == 0:
            continue

        # Create training example for each market
        for idx, market in weekly_markets.iterrows():
            days_until_resolution = (market['expected_expiration_time'] - cutoff).total_seconds() / 86400

            example = {
                # Metadata
                'cutoff_time': cutoff.isoformat(),
                'week_id': cutoff.strftime('%Y-W%U'),
                'ticker': market['ticker'],
                'event_ticker': market['event_ticker'],

                # Question/Context
                'title': market['title'],
                'subtitle': market['subtitle'],
                'category': market['category'],
                'rules_primary': market['rules_primary'],

                # Market features (as of cutoff time)
                # Note: price_at_cutoff uses last_price as proxy since historical snapshots unavailable
                'price_at_cutoff': market['last_price'],  # For difficulty gauge
                'last_price': market['last_price'],
                'last_price_dollars': market['last_price_dollars'],
                'yes_bid': market['yes_bid'],
                'yes_ask': market['yes_ask'],
                'no_bid': market['no_bid'],
                'no_ask': market['no_ask'],
                'volume': market['volume'],
                'volume_24h': market['volume_24h'],
                'open_interest': market['open_interest'],
                'liquidity': market['liquidity'],
                'liquidity_dollars': market['liquidity_dollars'],

                # Temporal features
                'days_until_resolution': days_until_resolution,
                'days_since_open': (cutoff - market['open_time']).total_seconds() / 86400,
                'market_duration_days': market['market_duration_days'],
                'open_time': market['open_time'].isoformat(),
                'expected_expiration_time': market['expected_expiration_time'].isoformat(),

                # Market structure
                'market_type': market['market_type'],
                'strike_type': market['strike_type'],
                'floor_strike': market['floor_strike'],

                # Label (ground truth outcome)
                'result': market['result'],
                'settlement_value': market['settlement_value'],
            }

            training_data.append(example)

            # Update stats
            week_key = cutoff.strftime('%Y-W%U')
            weekly_stats[week_key]['total_markets'] += 1
            if market['result'] == 'yes':
                weekly_stats[week_key]['yes_outcomes'] += 1
            else:
                weekly_stats[week_key]['no_outcomes'] += 1

    return training_data, weekly_stats

def save_training_data(training_data, output_dir):
    """Save training data as parquet"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(training_data)

    # Save as parquet (efficient for large datasets)
    parquet_path = output_dir / 'futurex_training_data.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet to {parquet_path}")

    return df

def print_statistics(df, weekly_stats):
    """Print comprehensive statistics about the training data"""
    print("\n" + "="*80)
    print("TRAINING DATA STATISTICS")
    print("="*80)

    print(f"\nTotal training examples: {len(df)}")
    print(f"Unique weeks: {df['week_id'].nunique()}")
    print(f"Unique markets: {df['ticker'].nunique()}")

    print("\n" + "-"*80)
    print("OUTCOME DISTRIBUTION:")
    print("-"*80)
    outcome_counts = df['result'].value_counts()
    print(outcome_counts)
    print(f"Yes percentage: {outcome_counts['yes']/len(df)*100:.1f}%")
    print(f"No percentage: {outcome_counts['no']/len(df)*100:.1f}%")

    print("\n" + "-"*80)
    print("PRICE DISTRIBUTION (at cutoff):")
    print("-"*80)
    print(df['last_price'].describe())

    print("\n" + "-"*80)
    print("FORECASTING HORIZON (days until resolution):")
    print("-"*80)
    print(df['days_until_resolution'].describe())

    print("\n" + "-"*80)
    print("TOP 10 WEEKS BY EXAMPLE COUNT:")
    print("-"*80)
    top_weeks = df['week_id'].value_counts().head(10)
    for week, count in top_weeks.items():
        stats = weekly_stats[week]
        print(f"{week}: {count} markets (Yes: {stats['yes_outcomes']}, No: {stats['no_outcomes']})")

    print("\n" + "-"*80)
    print("MARKET CATEGORIES:")
    print("-"*80)
    category_counts = df['category'].value_counts().head(10)
    print(category_counts)

    print("\n" + "-"*80)
    print("SAMPLE EXAMPLES:")
    print("-"*80)
    sample = df.sample(min(3, len(df)))
    for idx, row in sample.iterrows():
        print(f"\nWeek: {row['week_id']}")
        print(f"Title: {row['title'][:100]}...")
        print(f"Price at cutoff: {row['last_price']}¢")
        print(f"Days until resolution: {row['days_until_resolution']:.1f}")
        print(f"Actual outcome: {row['result']}")

    print("\n" + "="*80)

def main():
    """Main execution pipeline"""
    print("FutureX Training Data Preparation Pipeline")
    print("="*80)

    # Load data
    df = load_kalshi_data()

    # Filter for quality
    df_quality = filter_quality_markets(df)

    # Generate weekly cutoffs
    cutoffs = generate_weekly_cutoffs(df_quality)

    # Create training examples
    print("\nCreating training examples...")
    training_data, weekly_stats = create_training_examples(df_quality, cutoffs)
    print(f"Created {len(training_data)} training examples")

    # Save data
    print("\nSaving training data...")
    df_train = save_training_data(training_data, '/Users/dzorlu/src/kalshi-dataset/kalshi/data')

    # Print statistics
    print_statistics(df_train, weekly_stats)

    # Save metadata
    metadata = {
        'total_examples': len(training_data),
        'total_weeks': df_train['week_id'].nunique(),
        'date_generated': datetime.now().isoformat(),
        'config': {
            'cutoff_day': CUTOFF_DAY,
            'cutoff_hour_utc': CUTOFF_HOUR,
            'forecast_window_days': FORECAST_WINDOW_DAYS,
            'uncertainty_range': UNCERTAINTY_RANGE,
            'min_volume': MIN_VOLUME,
            'max_early_close_days': MAX_EARLY_CLOSE_DAYS,
        }
    }

    metadata_path = Path('/Users/dzorlu/src/kalshi-dataset/kalshi/data') / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")
    print("\nDone!")

if __name__ == '__main__':
    main()
