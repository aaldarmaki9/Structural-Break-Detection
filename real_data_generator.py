import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StockDataConfig1:
    """Configuration for stock-based structural break data"""
    n_series: int = 100
    series_length: int = 1000
    max_breaks: int = 3
    min_segment_length: int = 100
    start_date: str = "2000-01-01"
    end_date: str = "2017-12-31"
    seed: int = 42


# ============================================================
# MAIN DATA GENERATOR CLASS
# ============================================================

class RealStockDataGenerator:
    """
    Generate structural break data using real stock market returns
    by concatenating segments from different stocks.

    Important:
    - We do NOT attempt to screen segments for internal breaks.
    - Concatenation points are guaranteed breakpoints by construction.
    - Internal breaks inside segments may still exist (semi-synthetic setting).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.stock_data: Dict[str, np.ndarray] = {}

        # Diverse stock selection
        self.stock_symbols = [
            # High volatility tech stocks
            'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',

            # Low volatility defensive stocks
            'KO', 'PG', 'JNJ', 'WMT', 'XOM', 'CVX', 'PFE',

            # Financial stocks
            'JPM', 'BAC', 'WFC', 'GS', 'C',

            # Industrial/Cyclical stocks
            'CAT', 'BA', 'GE', 'MMM', 'HON',

            # Consumer discretionary
            'HD', 'MCD', 'DIS', 'SBUX', 'NKE',

            # Healthcare
            'UNH', 'ABBV', 'LLY', 'MRK'
        ]

        # Group stocks by characteristics
        self.stock_groups = {
            'high_vol_tech': ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'low_vol_defensive': ['KO', 'PG', 'JNJ', 'WMT', 'XOM', 'CVX', 'PFE'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'C'],
            'industrial': ['CAT', 'BA', 'GE', 'MMM', 'HON'],
            'consumer': ['HD', 'MCD', 'DIS', 'SBUX', 'NKE'],
            'healthcare': ['UNH', 'ABBV', 'LLY', 'MRK']
        }

    def download_stock_data(self, config: StockDataConfig1) -> bool:
        """Download stock data for all symbols and compute log-returns."""

        print(f"📈 DOWNLOADING REAL STOCK DATA")
        print("=" * 40)
        print(f"Period: {config.start_date} to {config.end_date}")
        print(f"Stocks: {len(self.stock_symbols)} symbols")

        successful_downloads = 0
        failed_downloads = []

        for i, symbol in enumerate(self.stock_symbols):
            try:
                print(f"  {symbol:<6} ({i+1:2d}/{len(self.stock_symbols):2d})", end=" ")

                # Download stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(start=config.start_date, end=config.end_date)

                if len(hist) < config.series_length * 2:
                    print(f"❌ Insufficient data ({len(hist)} days)")
                    failed_downloads.append(symbol)
                    continue

                # Calculate log returns
                hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))

                # Remove NaN values and extreme outliers
                returns = hist['Log_Return'].dropna()

                # Remove extreme outliers (beyond 5 std devs)
                mean_return = returns.mean()
                std_return = returns.std()
                returns = returns[np.abs(returns - mean_return) <= 5 * std_return]

                if len(returns) < config.series_length * 2:
                    print(f"❌ Insufficient clean data after outlier removal ({len(returns)} returns)")
                    failed_downloads.append(symbol)
                    continue

                # Store returns
                self.stock_data[symbol] = returns.values
                successful_downloads += 1
                print(f"✅ {len(returns)} returns")

            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}...")
                failed_downloads.append(symbol)

        print(f"\n📊 Download Summary:")
        print(f"  Successful: {successful_downloads}/{len(self.stock_symbols)} stocks")

        if failed_downloads:
            print(f"  Failed: {len(failed_downloads)} stocks")
            print(f"  Failed symbols: {failed_downloads}")

        # Update stock groups to remove failed downloads
        for group_name, symbols in self.stock_groups.items():
            self.stock_groups[group_name] = [s for s in symbols if s in self.stock_data]
            print(f"  {group_name}: {len(self.stock_groups[group_name])} stocks")

        return successful_downloads >= 15

    def _extract_segment(self,
                         stock_symbol: str,
                         segment_length: int) -> np.ndarray:
        """
        Extract a random segment of given length from the stock's returns.

        Note:
        - No attempt is made to check for internal breaks.
        - If there is not enough data, we take as much as possible from the start.
        """
        stock_returns = self.stock_data[stock_symbol]
        max_start = len(stock_returns) - segment_length

        if max_start <= 0:
            # Not enough data: truncate or pad if necessary
            return stock_returns[:segment_length]

        start_idx = self.rng.randint(0, max_start)
        return stock_returns[start_idx:start_idx + segment_length]

    def _generate_single_stock_series(
        self,
        config: StockDataConfig1
    ) -> Tuple[np.ndarray, List[int], str, List[str]]:
        """
        Generate a series from a single stock (no labelled breaks).

        This corresponds to the 'no_break' case: one segment, one stock.
        """
        available_stocks = list(self.stock_data.keys())
        stock_symbol = self.rng.choice(available_stocks)
        segment = self._extract_segment(stock_symbol, config.series_length)

        return segment, [], 'no_break', [stock_symbol]

    def _generate_multi_stock_series(
        self,
        config: StockDataConfig1,
        n_breaks: int
    ) -> Tuple[np.ndarray, List[int], str, List[str]]:
        """
        Generate a series from multiple stocks with structural breaks.

        Breaks are created only at concatenation points between segments
        coming from (potentially) different stocks / groups.

        Internal breaks inside segments may still exist (semi-synthetic).
        """

        # Choose break type
        break_types = [
            'volatility_shift',
            'sector_rotation',
            'economic_cycle',
            'market_regime',
            'cross_sector_random'
        ]
        break_type = self.rng.choice(break_types)

        # Generate segment boundaries
        n_segments = n_breaks + 1
        segment_lengths = self._generate_segment_lengths(
            config.series_length, n_segments, config.min_segment_length
        )

        # Choose stocks for each segment
        stock_symbols = self._choose_stocks_for_break_type(break_type, n_segments)

        # Generate series by concatenating segments
        series_parts: List[np.ndarray] = []
        break_points: List[int] = []
        current_position = 0

        for i, (segment_length, stock_symbol) in enumerate(zip(segment_lengths, stock_symbols)):
            segment = self._extract_segment(stock_symbol, segment_length)
            series_parts.append(segment)

            # Record break point after each segment except the last
            if i < n_segments - 1:
                current_position += segment_length
                break_points.append(current_position)

        series = np.concatenate(series_parts)

        return series, break_points, break_type, stock_symbols

    def _generate_segment_lengths(self,
                                  total_length: int,
                                  n_segments: int,
                                  min_length: int) -> List[int]:
        """Generate segment lengths that sum to total_length."""

        if n_segments * min_length > total_length:
            # Fallback: roughly equal segments
            base_length = total_length // n_segments
            lengths = [base_length] * n_segments
            remainder = total_length - sum(lengths)
            for i in range(remainder):
                lengths[i] += 1
            return lengths

        available_length = total_length - n_segments * min_length

        if available_length <= 0:
            return [min_length] * n_segments

        splits = np.sort(self.rng.choice(
            range(available_length + 1),
            n_segments - 1,
            replace=False
        ))
        splits = np.concatenate([[0], splits, [available_length]])

        lengths = []
        for i in range(n_segments):
            extra_length = splits[i + 1] - splits[i]
            lengths.append(min_length + extra_length)

        return lengths

    def _choose_stocks_for_break_type(self,
                                      break_type: str,
                                      n_segments: int) -> List[str]:
        """Choose stock symbols based on desired break type."""

        available_stocks = list(self.stock_data.keys())

        if break_type == 'volatility_shift':
            high_vol = [s for s in self.stock_groups['high_vol_tech'] if s in self.stock_data]
            low_vol = [s for s in self.stock_groups['low_vol_defensive'] if s in self.stock_data]

            symbols = []
            for i in range(n_segments):
                if i % 2 == 0 and high_vol:
                    symbols.append(self.rng.choice(high_vol))
                elif low_vol:
                    symbols.append(self.rng.choice(low_vol))
                else:
                    symbols.append(self.rng.choice(available_stocks))

        elif break_type == 'sector_rotation':
            sectors = ['high_vol_tech', 'financial', 'consumer', 'healthcare', 'industrial']
            available_sectors = [s for s in sectors if self.stock_groups.get(s)]

            symbols = []
            for i in range(n_segments):
                if available_sectors:
                    sector = available_sectors[i % len(available_sectors)]
                    sector_stocks = [s for s in self.stock_groups[sector] if s in self.stock_data]
                    if sector_stocks:
                        symbols.append(self.rng.choice(sector_stocks))
                    else:
                        symbols.append(self.rng.choice(available_stocks))
                else:
                    symbols.append(self.rng.choice(available_stocks))

        elif break_type == 'economic_cycle':
            cyclical = [s for s in self.stock_groups['industrial'] if s in self.stock_data]
            defensive = [s for s in (
                self.stock_groups['low_vol_defensive'] + self.stock_groups['healthcare']
            ) if s in self.stock_data]

            symbols = []
            for i in range(n_segments):
                if i % 2 == 0 and cyclical:
                    symbols.append(self.rng.choice(cyclical))
                elif defensive:
                    symbols.append(self.rng.choice(defensive))
                else:
                    symbols.append(self.rng.choice(available_stocks))

        elif break_type == 'market_regime':
            growth = [s for s in (
                self.stock_groups['high_vol_tech'] + self.stock_groups['consumer']
            ) if s in self.stock_data]
            value = [s for s in (
                self.stock_groups['financial'] + self.stock_groups['industrial']
            ) if s in self.stock_data]

            symbols = []
            for i in range(n_segments):
                if i % 2 == 0 and growth:
                    symbols.append(self.rng.choice(growth))
                elif value:
                    symbols.append(self.rng.choice(value))
                else:
                    symbols.append(self.rng.choice(available_stocks))

        else:  # cross_sector_random
            symbols = []
            used_stocks = set()

            for i in range(n_segments):
                available_choices = [s for s in available_stocks if s not in used_stocks]
                if not available_choices:
                    available_choices = available_stocks

                chosen_stock = self.rng.choice(available_choices)
                symbols.append(chosen_stock)
                used_stocks.add(chosen_stock)

        # Ensure we have enough symbols
        while len(symbols) < n_segments:
            symbols.append(self.rng.choice(available_stocks))

        return symbols[:n_segments]

    def generate_structural_break_dataset(self,
                                          config: StockDataConfig1) -> pd.DataFrame:
        """
        Generate dataset with structural breaks using real stock data.

        - Each series is a concatenation of 1 to max_breaks+1 segments.
        - Breakpoints occur only at concatenation points.
        - Internal breaks inside segments may exist (semi-synthetic).
        """

        print(f"\n🔧 GENERATING REAL-STOCK CONCATENATED DATASET")
        print("=" * 55)
        print(f"Target: {config.n_series} series of {config.series_length} returns each")

        if not self.stock_data:
            print("❌ No stock data available. Download data first.")
            return pd.DataFrame()

        results = []
        break_type_counts: Dict[str, int] = {}

        for series_id in range(config.n_series):
            if (series_id + 1) % 25 == 0:
                print(f"  Generated {series_id + 1}/{config.n_series} series...")

            # Determine number of breaks
            break_probs = [0.25, 0.40, 0.25, 0.10]  # for 0,1,2,3 breaks
            n_breaks = self.rng.choice(range(config.max_breaks + 1), p=break_probs)

            if n_breaks == 0:
                series, breaks, break_type, stock_info = self._generate_single_stock_series(config)
            else:
                series, breaks, break_type, stock_info = self._generate_multi_stock_series(
                    config, n_breaks
                )

            break_type_counts[break_type] = break_type_counts.get(break_type, 0) + 1

            # Create result record
            result: Dict[str, object] = {
                'series_id': series_id,
                'n_breaks': len(breaks),
                'break_points': breaks,
                'primary_break_type': break_type,
                'stock_symbols': stock_info,
                'data_source': 'real_stock_returns'
            }

            # Add time series data
            for t in range(len(series)):
                result[f't_{t}'] = series[t]

            results.append(result)

        dataset = pd.DataFrame(results)

        # Print summary
        print(f"\n📊 DATASET SUMMARY:")
        print("=" * 40)
        print(f"Total series: {len(dataset)}")

        break_counts = dataset['n_breaks'].value_counts().sort_index()
        print(f"\nBreak count distribution:")
        for n_breaks, count in break_counts.items():
            print(f"  {n_breaks} breaks: {count:3d} series ({count/len(dataset)*100:5.1f}%)")

        print(f"\nBreak type distribution:")
        for break_type, count in sorted(break_type_counts.items()):
            print(f"  {break_type:<20}: {count:3d} series ({count/len(dataset)*100:5.1f}%)")

        return dataset

    def save_dataset(self, dataset: pd.DataFrame, filename: str):
        """Save dataset to file."""
        dataset.to_pickle(filename)
        print(f"\n✅ Dataset saved to {filename}")


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def generate_real_stock_dataset(config: StockDataConfig1 = None,
                                cache_file: str = None) -> pd.DataFrame:
    """
    Generate real-stock concatenated dataset (semi-synthetic).

    - Downloads returns for a set of equities.
    - Builds time series by concatenating segments from different stocks.
    - Concatenation points are labelled as structural breaks.
    """

    if config is None:
        config = StockDataConfig1()

    # Check cache
    if cache_file:
        try:
            import os
            if os.path.exists(cache_file):
                print(f"📂 Loading cached dataset from {cache_file}")
                return pd.read_pickle(cache_file)
        except Exception:
            print("⚠️  Could not load cache, generating new dataset")

    # Generate new dataset
    generator = RealStockDataGenerator(config.seed)

    print("🚀 REAL-STOCK CONCATENATED DATASET GENERATION")
    print("=" * 55)

    if not generator.download_stock_data(config):
        print("❌ Failed to download sufficient stock data")
        return pd.DataFrame()

    dataset = generator.generate_structural_break_dataset(config)

    if cache_file and not dataset.empty:
        generator.save_dataset(dataset, cache_file)

    return dataset


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Configuration
    config = StockDataConfig1(
        n_series=100,
        series_length=1000,
        max_breaks=3,
        min_segment_length=100,
        start_date="2000-01-01",
        end_date="2017-12-31",
        seed=42
    )

    print("🚀 GENERATING REAL-STOCK CONCATENATED DATASET")
    print("=" * 55)
    print("Concatenation points are treated as structural breaks.")
    print("Internal breaks within segments may still exist (semi-synthetic setting).\n")

    # Generate dataset
    dataset = generate_real_stock_dataset(config, "real_stock_breaks.pkl")

    if not dataset.empty:
        print("\n✅ SUCCESS! Generated dataset")
        print(f"   Shape: {dataset.shape}")
        print(f"   Columns: {list(dataset.columns[:8])}...")

        # Show examples
        print("\n📊 Example Series:")
        for i in range(min(3, len(dataset))):
            row = dataset.iloc[i]
            print(f"\n  Series {i}:")
            print(f"    Breaks: {row['n_breaks']}")
            print(f"    Break points: {row['break_points']}")
            print(f"    Break type: {row['primary_break_type']}")
            print(f"    Stocks: {row['stock_symbols']}")
    else:
        print("❌ Failed to generate dataset")
