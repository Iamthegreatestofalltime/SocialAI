import pandas as pd
import json
import os

def excel_to_json(excel_path, json_path):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Convert DataFrame to list of dictionaries
    data = df.to_dict(orient='records')
    
    # Write to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    # Get the username from the user 
    username = input("Enter the Instagram username: ")
    
    # Construct the paths
    excel_path = os.path.join(username, f'{username}_post_details.xlsx')
    json_path = os.path.join(username, f'{username}_post_details.json')
    
    # Convert Excel to JSON
    excel_to_json(excel_path, json_path)
    
    print(f"JSON file has been created at: {json_path}")

if __name__ == "__main__":
    main()