#!/usr/bin/env python3
"""
Verification script for Phase 3 readiness.
Run this before starting Phase 3.

Usage:
    python verify_phase3_ready.py
"""
import sys
import subprocess
from pathlib import Path

def check(condition, message, fix=None):
    """Check a condition and print result."""
    if condition:
        print(f"✓ {message}")
        return True
    else:
        print(f"✗ {message}")
        if fix:
            print(f"  Fix: {fix}")
        return False

def main():
    print("="*60)
    print("PHASE 3 READINESS CHECK")
    print("="*60)
    
    all_checks = []
    
    # 1. Check Python packages
    print("\n1. Checking Python packages...")
    try:
        import pandas
        all_checks.append(check(True, "pandas installed"))
    except ImportError:
        all_checks.append(check(False, "pandas installed", "pip install pandas"))
    
    try:
        import pymc as pm
        all_checks.append(check(True, "pymc installed"))
    except ImportError:
        all_checks.append(check(False, "pymc installed", "pip install pymc==5.9.0"))
    
    try:
        import arviz as az
        all_checks.append(check(True, "arviz installed"))
    except ImportError:
        all_checks.append(check(False, "arviz installed", "pip install arviz"))
    
    try:
        import seaborn
        all_checks.append(check(True, "seaborn installed"))
    except ImportError:
        all_checks.append(check(False, "seaborn installed", "pip install seaborn"))
    
    try:
        import matplotlib
        all_checks.append(check(True, "matplotlib installed"))
    except ImportError:
        all_checks.append(check(False, "matplotlib installed", "pip install matplotlib"))
    
    try:
        import scipy
        all_checks.append(check(True, "scipy installed"))
    except ImportError:
        all_checks.append(check(False, "scipy installed", "pip install scipy"))
    
    try:
        import sqlalchemy
        all_checks.append(check(True, "sqlalchemy installed"))
    except ImportError:
        all_checks.append(check(False, "sqlalchemy installed", "pip install sqlalchemy"))
    
    try:
        import psycopg2
        all_checks.append(check(True, "psycopg2 installed"))
    except ImportError:
        all_checks.append(check(False, "psycopg2 installed", "pip install psycopg2-binary"))
    
    # 2. Check directory structure
    print("\n2. Checking directory structure...")
    dirs_to_check = [
        'notebooks',
        'notebooks/exploration',
        'notebooks/analysis',
        'docs',
        'models',
        'src/models',
        'tests/unit'
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        all_checks.append(check(path.exists(), f"{dir_path}/ exists", f"mkdir -p {dir_path}"))
    
    # 3. Check database connection
    print("\n3. Checking database...")
    try:
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://medhanshchoubey@localhost:5432/football_props")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT 1")).fetchone()
            all_checks.append(check(True, "Database connection works"))
            
            # Check player_features table
            result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM player_features")).fetchone()
            count = result[0]
            all_checks.append(check(count == 1720, f"player_features has {count} rows (expected 1720)"))
            
            # Check date range
            result = conn.execute(sqlalchemy.text(
                "SELECT MIN(match_date), MAX(match_date) FROM player_features"
            )).fetchone()
            min_date, max_date = result
            print(f"  Data range: {min_date} to {max_date}")
            all_checks.append(check(True, f"Date range verified"))
            
    except Exception as e:
        all_checks.append(check(False, "Database connection", str(e)))
    
    # 4. Check Jupyter
    print("\n4. Checking Jupyter...")
    try:
        import jupyter
        all_checks.append(check(True, "jupyter installed"))
    except ImportError:
        all_checks.append(check(False, "jupyter installed", "pip install jupyter"))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"Passed: {passed}/{total}")
    
    if all(all_checks):
        print("\n✅ ALL CHECKS PASSED - Ready for Phase 3!")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - Fix issues above before proceeding")
        return 1

if __name__ == '__main__':
    sys.exit(main())