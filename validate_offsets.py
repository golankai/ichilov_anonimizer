import json
import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def validate_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    failures = 0
    total_entities = 0
    for row in data:
        text = row['text']
        for ent in row['entities']:
            total_entities += 1
            start = ent['start']
            end = ent['end']
            expected_text = ent['text']
            
            sliced_text = text[start:end]
            if sliced_text != expected_text:
                print(f"Mismatch in row {row.get('id', 'N/A')}: expected '{expected_text}', got '{sliced_text}' at {start}:{end}")
                failures += 1
                
    print(f"Validated {total_entities} entities across {len(data)} rows.")
    if failures == 0:
        print("SUCCESS! All entity offsets are perfectly accurate and point to the exact character indices globally.")
    else:
        print(f"FAILED! Found {failures} offset mismatches.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate JSON entity offsets")
    parser.add_argument("--json", required=True, help="Path to output JSON")
    args = parser.parse_args()
    
    validate_json(args.json)
