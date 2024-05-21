### This script restructures the interleaved checkpoint to a regular pipeline parallel checkpoint format for inference.
### Resulting checkpoint will have the same PP and TP size as the original checkpoint.

from pathlib import Path
import torch
import re
from pathlib import Path
import argparse
import os
from tqdm import tqdm
import sys
sys.path.append('.')
import time

def log(*args, **kwargs):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp, *args, **kwargs)

def get_tp_model_list(checkpoint_path, tp_rank):
    glob_string = 'mp_rank_{:02d}*'.format(tp_rank)
    # get the list of pipeline shards per each tensor parallel rank
    path = sorted([pt.absolute() for pt in Path(checkpoint_path).glob(glob_string) if pt.is_dir()])
    tp_model_list = []

    for shard_path in path:
        shard = torch.load(str(shard_path) + '/model_optim_rng.pt', map_location='cpu')
        tp_model_list.append(shard)
    return tp_model_list


def calculate_real_layer_idx(layer_name, pp_rank, vp_rank, pipeline_parallel_model_size, num_layers_per_virtual_pipeline_stage):
    vertical_pp_size = pipeline_parallel_model_size * num_layers_per_virtual_pipeline_stage
    real_pp_rank = pp_rank * num_layers_per_virtual_pipeline_stage + vertical_pp_size * vp_rank
    # print("layer name: ", layer_name)
    layer_idx = int(re.search(r"\.(\d*)\.", layer_name).groups()[0])
    real_layer_idx = real_pp_rank + layer_idx
    return real_layer_idx

def calculate_new_pp_rank(real_layer_idx, num_layers, pipeline_parallel_model_size):
    # Calculates the new location for the num of layers in a vp model
    layers_per_pp_stage = num_layers // pipeline_parallel_model_size
    new_pp_rank = real_layer_idx % layers_per_pp_stage
    return new_pp_rank

def unstructure_pipeline_parallel_model(model_list, virtual_pipeline_parallel_size, pipeline_parallel_model_size, num_layers_per_virtual_pipeline_stage):
    unstructured_model = {'model': {'language_model': {'encoder':{}}}}
    layer_regex = re.compile(r"\.(\d*)\.")
    
    final_norm_layer = None
    # Encoder layers
    for model_idx, model in enumerate(model_list):
        for vp_model_idx in range(virtual_pipeline_parallel_size):
            encoder = model[f'model{vp_model_idx}']['language_model']['encoder']
            for layer_name in encoder.keys():
                if 'final_norm.weight' in layer_name:
                    final_norm_layer = encoder[layer_name]
                    continue
                else:
                    real_layer_idx = calculate_real_layer_idx(layer_name, 
                                                          pp_rank=model_idx, 
                                                          vp_rank=vp_model_idx, 
                                                          pipeline_parallel_model_size=pipeline_parallel_model_size, 
                                                          num_layers_per_virtual_pipeline_stage=num_layers_per_virtual_pipeline_stage
                                                          )
                    new_name = layer_regex.sub(f".{str(real_layer_idx)}.", layer_name)
                unstructured_model['model']['language_model']['encoder'][new_name] = encoder[layer_name]
    
    # Sort encoder layers enumerically 
    reorder_layers = lambda model: {
        k: v for k, v in sorted(model['model']['language_model']['encoder'].items(), 
        key=lambda item: int(re.search(r"\.(\d*)\.", item[0]).groups()[0])
        )}

    unstructured_model['model']['language_model']['encoder'] = reorder_layers(unstructured_model)
    
    # Add final norm layer
    unstructured_model['model']['language_model']['encoder'][f"final_norm.weight"] = final_norm_layer
    # Add embedding layer 
    unstructured_model['model']['language_model']['embedding'] = model_list[0]['model0']['language_model']['embedding']
    # Add output layer
    unstructured_model['model']['language_model']['output_layer'] = model_list[-1]['model4']['language_model']['output_layer']
    
    # Add arguments 
    unstructured_model['args'] = model_list[0]['args']
    unstructured_model['checkpoint_version'] = model_list[0]['checkpoint_version'] 
    
    return unstructured_model

def restructure_keys(model, num_layers, new_pipeline_parallel_model_size):
    pipeline_shards = []
    final_norm_layer = None
    for i in range(new_pipeline_parallel_model_size):
        pipeline_shards.append({'model': {'language_model': {'encoder':{}}}})
    encoder = model['model']['language_model']['encoder']
    layer_num_regex = re.compile(r"\.(\d*)\.")
    for layer_name in encoder.keys():
        if layer_name == 'final_norm.weight':
            final_norm_layer = encoder[layer_name]
        else:
            layer_idx = int(layer_num_regex.search(layer_name).groups()[0])
            new_idx_in_shard = calculate_new_pp_rank(layer_idx, num_layers, new_pipeline_parallel_model_size)
            shard_idx = layer_idx // (num_layers // new_pipeline_parallel_model_size)
            new_layer_name = layer_num_regex.sub(f".{str(new_idx_in_shard)}.", layer_name)
            pipeline_shards[shard_idx]['model']['language_model']['encoder'][new_layer_name] = encoder[layer_name]
            pipeline_shards[shard_idx]['args'] = model['args']
            pipeline_shards[shard_idx]['checkpoint_version'] = model['checkpoint_version']
 
    pipeline_shards[0]['model']['language_model']['embedding'] = model['model']['language_model']['embedding']
    pipeline_shards[-1]['model']['language_model']['encoder'][f"final_norm.weight"] = final_norm_layer
    pipeline_shards[-1]['model']['language_model']['output_layer'] = model['model']['language_model']['output_layer']

    return pipeline_shards



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--tp_size', type=int, default=8, help='tensor model parallel size')
    parser.add_argument('--pp_size', type=int, default=8, help='pipeline model parallel size')
    parser.add_argument('--vpp_size', type=int, default=5, help='virtual pipeline model parallel size')
    parser.add_argument('--new_pp_size', type=int, default=None, help='new pipeline model parallel size')
    parser.add_argument('--num_layers_per_vpp_stage', type=int, default=2, help='number of layers per virtual pipeline stage')
    parser.add_argument('--num_layers', type=int, default=80, help='number of layers in the model')
    parser.add_argument('--output_path', type=str, required=True, help='output path for the restructured model')

    args = parser.parse_args()
    if args.new_pp_size is None:
        args.new_pp_size = args.pp_size

    log(args)
    for tp in tqdm(range(args.tp_size)):
        models = get_tp_model_list(args.checkpoint_path, tp)
        combinded_model = unstructure_pipeline_parallel_model(models, args.vpp_size, args.pp_size, args.num_layers_per_vpp_stage)
        restructured_pp_shards = restructure_keys(combinded_model, num_layers=args.num_layers, new_pipeline_parallel_model_size=args.new_pp_size)
        log("Saving:", end=' ')

        for i, shard in enumerate(restructured_pp_shards):
            shard_dir_name = f"mp_rank_{tp:02d}_{i:03d}"
            save_path = os.path.join(args.output_path, shard_dir_name, 'model_optim_rng.pt')
            os.makedirs(os.path.join(args.output_path, shard_dir_name), exist_ok=True)
            print(f"{shard_dir_name}", end=', ')
            torch.save(shard, save_path)
        print()

    log("Conversion done!")

        

if __name__ == '__main__':
    main()