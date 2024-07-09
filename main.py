import sys
import pandas as pd
from intermediate_template import return_mid
def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <df_path> <condition>")
        sys.exit(1)
    
    df_path = sys.argv[1]
    condition = sys.argv[2]
    
    try:
        df = pd.read_csv(df_path)
    except Exception as e:
        print(f"Error reading the file {df_path}: {e}")
        sys.exit(1)
    df['condtion'] = condition
    df['mid'] = df.apply(return_mid, axis=1)
    df.to_csv("processed_df.csv", encoding='utf-8')

if __name__ == "__main__":
    main()
