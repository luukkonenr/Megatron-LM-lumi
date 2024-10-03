import argparse
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm 

def compare_parameters(m1, m2):
	size = len(m2.model.layers.state_dict().keys())
	differing_keys = []
	for a1, a2 in tqdm(zip(m1.model.layers.state_dict().items(), m2.model.layers.state_dict().items()), total=size):
		if not (torch.allclose(a1[1],a2[1])):
			print(f'key difference in {a1[0]}')
			differing_keys.append(a1[0])
	print(f"Num differing keys: {len(differing_keys)}")
	print(differing_keys)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=str)
    parser.add_argument("output_model", type=str)
    args = parser.parse_args()
    m1 = AutoModelForCausalLM.from_pretrained(args.input_model)
    m2 = AutoModelForCausalLM.from_pretrained(args.output_model)
    compare_parameters(m1,m2)
    print("Done.")
	
    
if __name__ == '__main__':
    main()
