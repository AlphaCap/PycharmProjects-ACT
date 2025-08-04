"""
Run Sector Optimization with 2-Year Dataset
Optimized for shorter time periods but still effective
"""

from sector_etf_optimizer import SectorETFOptimizer
from sector_parameter_manager import SectorParameterManager
import pandas as pd

def run_2year_optimization():
    print("🎯 Running Sector Optimization with 2-Year Data")
    print("=" * 50)
    print("📅 Using data: 2023-07-24 to 2025-07-21 (2.0 years)")
    print("🔧 Adjusted for shorter walk-forward periods")
    
    # Initialize optimizer with settings optimized for 2-year data
    optimizer = SectorETFOptimizer(optimization_mode="fast")
    
    # Adjust walk-forward periods for 2-year dataset
    optimizer.train_months = 6  # 6 months training 
    optimizer.test_months = 2   # 2 months testing 
    optimizer.min_data_points = 80  # Lower minimum (was 100)
    
    # This gives us ~4 walk-forward periods with 24 months of data
    
    print(f"🔄 Walk-forward settings:")
    print(f"   📚 Training period: {optimizer.train_months} months")
    print(f"   🧪 Testing period: {optimizer.test_months} months")
    print(f"   📊 Minimum data points: {optimizer.min_data_points}")
    
    # Show available ETFs
    print(f"\n📊 Available ETFs for optimization:")
    for sector, etf in optimizer.param_manager.sector_etfs.items():
        print(f"   {etf}: {sector}")
    
    # Run optimization
    choice = input(f"\nRun optimization on all 11 sectors? (y/N): ").strip().lower()
    
    if choice == 'y':
        print(f"\n🚀 Starting optimization...")
        print(f"⏱️  Estimated time: 5-15 minutes (depending on mode)")
        
        try:
            results = optimizer.optimize_all_sectors()
            
            if results:
                print(f"\n✅ Optimization Complete!")
                print(f"📊 Successfully optimized {len(results)} sectors")
                
                # Show brief results
                for sector, result in results.items():
                    perf = result['performance']
                    print(f"   {sector}: {perf['avg_test_roi']:.2f}% ROI, {perf['total_trades']} trades")
                
                # Show parameter summary
                print(f"\n📋 Parameter Summary:")
                param_manager = SectorParameterManager()
                summary = param_manager.get_optimization_summary()
                print(summary.to_string(index=False))
                
                print(f"\n✅ Ready to run backtests with optimized parameters!")
                print(f"💡 Next step: Test with ngs_sector_adapter.py")
                
            else:
                print(f"❌ No sectors were successfully optimized")
                
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Test single sector
        print(f"\n🧪 Testing single sector optimization (Technology/XLK)...")
        try:
            result = optimizer.optimize_etf_parameters('XLK', 'Technology')
            if result:
                print(f"✅ Technology sector optimization successful!")
                print(f"   Best ROI: {result['performance']['avg_test_roi']:.2f}%")
                print(f"   Trades: {result['performance']['total_trades']}")
            else:
                print(f"❌ Technology optimization failed")
        except Exception as e:
            print(f"❌ Error: {e}")

def check_data_quality():
    """Quick check of the 2-year data quality"""
    print(f"\n🔍 Data Quality Check:")
    
    from etf_historical_downloader import ETFHistoricalDownloader
    downloader = ETFHistoricalDownloader()
    summary = downloader.get_download_summary()
    
    available_data = summary[summary['Status'] == 'Available']
    
    print(f"📊 Data Summary:")
    print(f"   ETFs with data: {len(available_data)}/11")
    print(f"   Average records: {available_data['Records'].mean():.0f}")
    print(f"   Date range: {available_data['Start Date'].iloc[0]} to {available_data['End Date'].iloc[0]}")
    print(f"   Average years: {available_data['Years'].mean():.1f}")
    
    if len(available_data) >= 8 and available_data['Records'].mean() >= 400:
        print(f"✅ Data quality is good for optimization!")
        return True
    else:
        print(f"⚠️  Data quality may be insufficient")
        return False

if __name__ == "__main__":
    # Check data first
    data_ok = check_data_quality()
    
    if data_ok:
        run_2year_optimization()
    else:
        print(f"\n💡 Consider getting more data or using Yahoo Finance alternative")