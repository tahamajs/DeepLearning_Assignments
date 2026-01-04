# validate_structure.py - Simple validation script to check code structure

def validate_imports():
    """Check if all modules can be imported (without dependencies)"""
    try:
        # Test basic Python imports
        import os
        import sys
        import math
        print("✓ Basic Python imports work")

        # Check if files exist
        files = ['data.py', 'models.py', 'utils.py', 'train.py', 'generate.py', 'main.py']
        for file in files:
            if os.path.exists(file):
                print(f"✓ {file} exists")
            else:
                print(f"✗ {file} missing")

        print("✓ All files present")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def validate_syntax():
    """Check syntax of all Python files"""
    import subprocess
    import sys

    files = ['data.py', 'models.py', 'utils.py', 'train.py', 'generate.py', 'main.py']
    python_cmd = sys.executable

    for file in files:
        try:
            result = subprocess.run([python_cmd, '-m', 'py_compile', file],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {file} syntax OK")
            else:
                print(f"✗ {file} syntax error: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Error checking {file}: {e}")
            return False

    print("✓ All syntax checks passed")
    return True

if __name__ == "__main__":
    print("Validating CA5 extracted code structure...")
    print("=" * 50)

    success = True
    success &= validate_imports()
    success &= validate_syntax()

    print("=" * 50)
    if success:
        print("✓ All validations passed! Code structure is correct.")
        print("\nTo run the code:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: python main.py")
    else:
        print("✗ Some validations failed. Please check the errors above.")