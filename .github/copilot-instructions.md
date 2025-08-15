# Python Trading System & AI Platform
**PycharmProjects-ACT** is a comprehensive Python-based algorithmic trading system featuring multiple AI-driven strategies, real-time market data integration, and advanced portfolio management capabilities. The system includes a Streamlit-based web dashboard, multiple trading strategies (nGS, AlphaCaptureAI, gSTDayTrader), and sophisticated data management tools.

**ALWAYS** reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository
- **Install Dependencies** (takes ~3 minutes):
  ```bash
  python -m pip install -r requirements.txt
  python -m pip install pytest black isort flake8 mypy bandit pytest-cov
  ```
- **Run Tests** (takes ~3 seconds):
  ```bash
  python -m pytest tests/ -v
  ```
  - **NEVER CANCEL**: Test suite is fast but comprehensive
  - Expected: 6 tests pass, covers data loading and structure validation
- **Code Formatting and Linting** (takes ~1-2 seconds each):
  ```bash
  python -m black .          # Format code (1 second)
  python -m isort .           # Sort imports (0.2 seconds)  
  python -m flake8 . --count  # Check style (0.7 seconds)
  ```
  - **Note**: flake8 reports ~347 style violations but does not break functionality

### Run the Application
- **Start Main Streamlit App** (takes ~2 seconds to start):
  ```bash
  streamlit run app.py --server.port 8501 --server.headless true
  ```
  - **NEVER CANCEL**: App starts quickly and runs continuously
  - Access at `http://localhost:8501`
  - **Multi-page navigation**: Main dashboard → nGS System → AlphaCaptureAI → gSTDayTrader → admin dashboard

### Core Module Testing
- **Test Data Manager** (0.7 seconds):
  ```bash
  python -c "from data_manager import initialize; initialize(); print('Data manager ready')"
  ```
- **Test AI Components** (0.4 seconds each):
  ```bash
  python -c "from comprehensive_indicator_library import ComprehensiveIndicatorLibrary; lib = ComprehensiveIndicatorLibrary(); print('18 indicators loaded')"
  python -c "from performance_objectives import ObjectiveManager; mgr = ObjectiveManager(); print('5 objectives loaded')"
  ```

## Validation

### Application Scenarios
- **ALWAYS** test complete user workflows after making changes:
  1. **Main Dashboard**: Verify portfolio metrics, positions table, and signals display correctly
  2. **nGS System**: Test historical performance, trade history, and portfolio analytics
  3. **Navigation**: Ensure all page transitions work (main → nGS System → AlphaCaptureAI → admin)
  4. **Data Loading**: Confirm data manager loads 66 positions and 502 S&P 500 symbols
- **Manual UI Testing**: Run `streamlit run app.py` and manually navigate through all pages
- **API Requirements**: gSTDayTrader page requires POLYGON_API_KEY configuration (shows import error without it)
- **Admin Access**: Admin dashboard requires password authentication ("4250Galt" hardcoded)

### Performance Validation
- **Data Manager Initialization**: ~0.7 seconds, loads 502 S&P 500 symbols and 66 positions
- **Streamlit App Startup**: ~2 seconds from command to accessible web interface
- **Test Suite**: 6 tests complete in ~3 seconds with 84% coverage on test files
- **Linting**: black (1s), isort (0.2s), flake8 (0.7s) - fast feedback cycle

## Known Issues and Workarounds

### Expected Issues
- **Network Connectivity**: Wikipedia sector data fetch fails (uses cached fallback) - NORMAL
- **API Configuration**: gSTDayTrader shows ImportError for POLYGON_API_KEY when not configured - EXPECTED
- **AlphaCaptureAI**: Shows "Coming Soon" placeholder - UNDER DEVELOPMENT
- **Style Violations**: flake8 reports ~347 violations but functionality works correctly

### Build Workarounds
- **Import Errors**: Some circular import warnings in AI modules - non-breaking
- **Missing Modules**: Some advanced features require additional setup (Polygon API) but core functionality works
- **Syntax Errors**: Fixed critical f-string issue in strategy_generator_ai.py during validation

## Common Tasks

### Development Workflow
- **Before Changes**: Always run `python -m pytest tests/` to establish baseline
- **After Changes**: Run `python -m black .` and `python -m isort .` then test app startup
- **Final Check**: Start Streamlit app and manually test navigation between all pages

### Auto-Repair System
- **Run Auto-Debug**: `python auto_debug_repair.py` (takes ~15 seconds)
  - Fixes code formatting, navigation issues, generates dynamic tests
  - **Note**: May produce test syntax errors in auto-generated files - ignore these

### Repository Structure
```
Project Root:
├── app.py                          # Main Streamlit dashboard
├── pages/                          # Multi-page Streamlit components
│   ├── 1_nGS_System.py            # Historical performance & analytics
│   ├── 2_AlphaCaptureAI.py        # Daily trading system (placeholder)
│   ├── 3_gSTDayTrader.py          # Day trading system (needs API config)
│   └── admin_dashboard.py         # Admin controls & debug tools
├── nGS_Revised_Strategy.py        # Core trading strategy implementation
├── data_manager.py                # Portfolio & market data management
├── comprehensive_indicator_library.py  # 18 technical indicators
├── strategy_generator_ai.py       # AI strategy generation system
├── requirements.txt               # Python dependencies
├── tests/unit/                    # Test suite (6 tests)
└── data/                          # Market data & trade history
```

### Key Data Files
- **S&P 500 Symbols**: `data/sp500_symbols.txt` (502 symbols loaded)
- **Trade History**: `data/trade_history.csv` (13,314 historical trades)
- **Positions**: Current portfolio positions (66 active positions)
- **ETF Data**: `data/etf_historical/` (sector ETF price data)

### API Integration
- **Polygon API**: Required for gSTDayTrader live data (set POLYGON_API_KEY environment variable)
- **Data Retention**: 6-month (180-day) rolling window for trade and position data
- **Sector Classification**: Falls back to static mapping when Wikipedia is unavailable

## Timing Expectations
- **Dependencies Install**: 3 minutes (pip install -r requirements.txt)
- **Test Suite**: 3 seconds (6 tests, good coverage)
- **Code Formatting**: 1 second (black), 0.2 seconds (isort)
- **Linting Check**: 0.7 seconds (flake8)
- **App Startup**: 2 seconds (streamlit run app.py)
- **Core Module Loading**: 0.4-0.7 seconds per module
- **Auto-Repair Cycle**: 15 seconds (comprehensive fixes)

**CRITICAL**: Never cancel any of these operations - they all complete quickly and provide essential feedback for development.