import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_file", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    print(args)
    with open(args.tokenizer_file, 'r') as f:
        tokenizer = json.load(f)['model']
        vocab = tokenizer['vocab']
        merges = tokenizer['merges']
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f, ensure_ascii=True)
        
        with open('merges.txt', 'w') as f:
            f.write('# This line is a required header')
            f.write('\n'.join(merges))

if __name__ == '__main__':
    main()