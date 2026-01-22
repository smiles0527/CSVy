"""
Interactive ML Pipeline with User Confirmations
Previews all operations before executing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate


class InteractivePipeline:
    """Pipeline with confirmation prompts for safety."""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(data_path)
        self.original_shape = self.df.shape
        print(f"\n{'='*80}")
        print(f"LOADED: {data_path}")
        print(f"Original size: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"{'='*80}\n")
    
    def preview_missing_data(self):
        """Show columns with missing data."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        has_missing = missing[missing > 0]
        if has_missing.empty:
            print("No missing values found")
            return []
        
        print("\nMISSING DATA SUMMARY:")
        print("-" * 60)
        
        data = []
        for col in has_missing.index:
            data.append([
                col,
                missing[col],
                f"{missing_pct[col]}%",
                "DROP" if missing_pct[col] > 50 else "KEEP"
            ])
        
        print(tabulate(data, headers=["Column", "Missing", "Percent", "Action"], tablefmt="simple"))
        
        # High missing columns
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            print(f"\n[WARNING] {len(high_missing)} columns have >50% missing:")
            for col in high_missing[:5]:
                print(f"  - {col}: {missing_pct[col]}%")
            if len(high_missing) > 5:
                print(f"  ... and {len(high_missing) - 5} more")
        
        return high_missing
    
    def preview_categorical_columns(self):
        """Show categorical columns and encoding preview."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            print("\nNo categorical columns found")
            return []
        
        print(f"\nCATEGORICAL COLUMNS: {len(cat_cols)}")
        print("-" * 60)
        
        data = []
        will_drop = []
        
        for col in cat_cols:
            nunique = self.df[col].nunique()
            action = "One-hot encode" if nunique <= 10 else "DROP (too many)"
            
            if nunique > 10:
                will_drop.append(col)
            
            data.append([
                col,
                nunique,
                str(self.df[col].mode()[0])[:30] if len(self.df[col].mode()) > 0 else "N/A",
                action
            ])
        
        print(tabulate(data, headers=["Column", "Unique Values", "Most Common", "Action"], tablefmt="simple"))
        
        if will_drop:
            print(f"\n[WARNING] {len(will_drop)} columns will be DROPPED (>10 unique values):")
            for col in will_drop[:5]:
                print(f"  - {col}")
            if len(will_drop) > 5:
                print(f"  ... and {len(will_drop) - 5} more")
        
        return will_drop
    
    def preview_cleaning_summary(self):
        """Show full summary of what will happen."""
        print("\n" + "="*80)
        print("CLEANING PREVIEW - WHAT WILL HAPPEN:")
        print("="*80)
        
        # Missing data
        high_missing = self.preview_missing_data()
        
        # Categorical columns
        cat_drops = self.preview_categorical_columns()
        
        # Final summary
        print("\n" + "="*80)
        print("SUMMARY:")
        print("-" * 60)
        
        total_drops = len(high_missing) + len(cat_drops)
        remaining = self.df.shape[1] - total_drops
        
        print(f"Original columns:       {self.df.shape[1]}")
        print(f"Drop (>50% missing):   -{len(high_missing)}")
        print(f"Drop (too categorical): -{len(cat_drops)}")
        print(f"Remaining:              {remaining}")
        print("="*80)
        
        return high_missing, cat_drops
    
    def confirm_cleaning(self):
        """Ask user to confirm cleaning operations."""
        high_missing, cat_drops = self.preview_cleaning_summary()
        
        if not high_missing and not cat_drops:
            print("\n[OK] No problematic columns found. Safe to proceed.")
            return True
        
        print("\n" + "="*80)
        response = input("Proceed with cleaning? (yes/no/customize): ").strip().lower()
        
        if response == 'yes' or response == 'y':
            return True
        elif response == 'customize' or response == 'c':
            return self.customize_cleaning(high_missing, cat_drops)
        else:
            print("[CANCELLED] No changes made.")
            return False
    
    def customize_cleaning(self, high_missing, cat_drops):
        """Let user choose what to keep/drop."""
        print("\nCUSTOMIZE CLEANING:")
        print("-" * 60)
        
        # High missing columns
        keep_missing = []
        if high_missing:
            print(f"\n{len(high_missing)} columns have >50% missing.")
            for col in high_missing:
                keep = input(f"  Keep '{col}'? (y/n): ").strip().lower()
                if keep == 'y':
                    keep_missing.append(col)
        
        # Categorical columns
        keep_cats = []
        if cat_drops:
            print(f"\n{len(cat_drops)} categorical columns have >10 unique values.")
            for col in cat_drops:
                keep = input(f"  Keep '{col}'? (y/n): ").strip().lower()
                if keep == 'y':
                    keep_cats.append(col)
        
        print(f"\n[OK] Will keep: {len(keep_missing) + len(keep_cats)} columns")
        print(f"[OK] Will drop: {len(high_missing) - len(keep_missing) + len(cat_drops) - len(keep_cats)} columns")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        return confirm == 'y'
    
    def run_with_confirmation(self):
        """Run full pipeline with user confirmation at each step."""
        if not self.confirm_cleaning():
            print("\n[STOPPED] Pipeline cancelled by user.")
            return None
        
        print("\n[OK] Running automated pipeline...")
        print("(Use automated_model_selection.py for actual training)")
        
        return self.df


def main():
    """Run interactive pipeline."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python interactive_pipeline.py <data.csv>")
        print("\nExample:")
        print("  python interactive_pipeline.py ../../data/hockey_data.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    pipeline = InteractivePipeline(data_path)
    result = pipeline.run_with_confirmation()
    
    if result is not None:
        print("\n[SUCCESS] Ready to train models.")
        print("Next step: python training/automated_model_selection.py")


if __name__ == "__main__":
    main()
