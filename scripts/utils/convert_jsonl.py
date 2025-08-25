#!/usr/bin/env python3
"""
Convert between pretty-printed JSON and regular JSONL formats.
"""

import json
import argparse
from pathlib import Path

def pretty_to_jsonl(input_file, output_file=None):
    """Convert pretty-printed JSON to regular JSONL."""
    if output_file is None:
        output_file = input_file.with_suffix('.jsonl')
    
    with open(input_file) as f:
        content = f.read().strip()
    
    # Split on }{ patterns for pretty-printed JSON
    entries = content.split('}\n{')
    
    with open(output_file, 'w') as f:
        for i, entry in enumerate(entries):
            if i > 0:
                entry = '{' + entry
            if i < len(entries) - 1:
                entry = entry + '}'
            if entry.strip():
                # Parse and write as single line
                data = json.loads(entry)
                f.write(json.dumps(data) + '\n')
    
    print(f"✅ Converted {len(entries)} entries to {output_file}")

def jsonl_to_pretty(input_file, output_file=None):
    """Convert regular JSONL to pretty-printed JSON."""
    if output_file is None:
        output_file = input_file.with_suffix('.pretty.jsonl')
    
    with open(input_file) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    with open(output_file, 'w') as f:
        for i, line in enumerate(lines):
            if i > 0:
                f.write('\n')  # Separator between entries
            data = json.loads(line)
            f.write(json.dumps(data, indent=2))
    
    print(f"✅ Converted {len(lines)} entries to pretty format: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert between JSONL formats')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--to-jsonl', action='store_true', 
                       help='Convert pretty JSON to regular JSONL')
    parser.add_argument('--to-pretty', action='store_true',
                       help='Convert regular JSONL to pretty JSON')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    
    if args.to_jsonl:
        pretty_to_jsonl(input_path, output_path)
    elif args.to_pretty:
        jsonl_to_pretty(input_path, output_path)
    else:
        # Auto-detect format
        with open(input_path) as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('{\n') or '\n  ' in first_line:
            print("Detected pretty-printed format, converting to JSONL...")
            pretty_to_jsonl(input_path, output_path)
        else:
            print("Detected JSONL format, converting to pretty...")
            jsonl_to_pretty(input_path, output_path)

if __name__ == "__main__":
    main()