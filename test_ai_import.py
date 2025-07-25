# test_ai_import.py
print("Testing AI module imports...")

try:
    print("1. Testing ngs_ai_integration_manager...")
    from ngs_ai_integration_manager import NGSAIIntegrationManager
    print("✅ NGSAIIntegrationManager imported successfully")
    
    print("2. Testing ngs_ai_performance_comparator...")
    from ngs_ai_performance_comparator import NGSAIPerformanceComparator  
    print("✅ NGSAIPerformanceComparator imported successfully")
    
    print("3. Testing object creation...")
    manager = NGSAIIntegrationManager(account_size=100000)
    comparator = NGSAIPerformanceComparator(account_size=100000)
    print("✅ AI objects created successfully")
    
    print("\n🎉 ALL AI MODULES WORKING!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("AI modules are not properly set up")
except Exception as e:
    print(f"❌ Other Error: {e}")
    print("AI modules imported but failed to initialize")
