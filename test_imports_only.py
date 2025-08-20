#!/usr/bin/env python3
"""
Lightweight test to check if the code structure is correct without heavy dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that the expected files exist."""
    print("🧪 Testing file structure...")
    
    expected_files = [
        "tree_seg/models/dinov3_adapter.py",
        "tree_seg/core/types.py",
        "tree_seg/models/initialization.py", 
        "tree_seg/models/preprocessing.py",
        "dinov3/hubconf.py",
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected files found")
    return True

def test_removed_files():
    """Test that old files were properly removed."""
    print("\n🧪 Testing removed files...")
    
    removed_files = [
        "tree_seg/core/upsampler.py",
        "tree_seg/core/patch.py",
        "tree_seg/utils/transform.py",
    ]
    
    still_present = []
    for file_path in removed_files:
        full_path = project_root / file_path
        if full_path.exists():
            still_present.append(file_path)
        else:
            print(f"✅ Removed: {file_path}")
    
    if still_present:
        print(f"❌ Files still present: {still_present}")
        return False
    
    print("✅ All old files properly removed")
    return True

def test_code_syntax():
    """Test that the Python files have valid syntax."""
    print("\n🧪 Testing code syntax...")
    
    test_files = [
        "tree_seg/models/dinov3_adapter.py",
        "tree_seg/core/types.py", 
        "tree_seg/models/initialization.py",
        "tree_seg/models/preprocessing.py",
    ]
    
    for file_path in test_files:
        try:
            full_path = project_root / file_path
            with open(full_path, 'r') as f:
                code = f.read()
            
            # Compile the code to check syntax
            compile(code, file_path, 'exec')
            print(f"✅ Syntax OK: {file_path}")
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False
    
    print("✅ All files have valid syntax")
    return True

def main():
    """Run lightweight tests."""
    print("🚀 DINOv3 Integration - Lightweight Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_removed_files,
        test_code_syntax,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n📊 Test Results:")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All lightweight tests passed! Code structure looks good.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -e .")
        print("   2. Run full tests with actual models")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)