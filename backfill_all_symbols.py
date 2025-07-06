from data_update import download_data_parallel, get_sp500_symbols, CONFIG
import time

# Set the number of days of history you want (e.g., 1000 for ~4 years, 2000 for ~8 years)
CONFIG["history_days"] = 200

# Optionally adjust batch size and workers for your system's capacity
CONFIG["batch_size"] = 50    # Number of symbols to process at once
CONFIG["max_workers"] = 8    # Number of threads (adjust for your hardware)

symbols = get_sp500_symbols()
print(f"Starting backfill for {len(symbols)} symbols, {CONFIG['history_days']} days each.")

# Download in batches to avoid API rate limits and memory issues
for i in range(0, len(symbols), CONFIG["batch_size"]):
    batch = symbols[i:i + CONFIG["batch_size"]]
    print(f"Processing batch {i // CONFIG['batch_size'] + 1} ({len(batch)} symbols)...")
    download_data_parallel(batch)
    print(f"Batch {i // CONFIG['batch_size'] + 1} complete.")
    time.sleep(2)  # Optional: Pause between batches to avoid rate limits

print("Backfill complete.")
