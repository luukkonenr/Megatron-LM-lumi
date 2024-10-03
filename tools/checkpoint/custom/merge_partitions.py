import sys
from pathlib import Path
import torch
from collections import OrderedDict
import re
import argparse
import os
from datetime import datetime

## Assumptions: 
# swiglu
# rotary positional embeddings
# untied embeddings
# no bias (for bias just add '.bias' in addition to '.weight')

# Search in directory above this
sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__).split("tools")[0])
)
# sys.path.append(os.path.abspath(
#     os.path.join(os.path.dirname(__file__),
#                     os.path.pardir,)))

PARALLEL_RANK_PATTERN = re.compile("mp_rank_(\d*)_?(\d*)?")
CP_ID_PATTERN = re.compile("iter_\d*")

DEVICE = 'cpu'
MODES = {'default':0, 'debug':1}
CUR_MODE = 'default'

def log(s, mode='default'):
    if MODES[mode]<=MODES[CUR_MODE]:
        now = str(datetime.now())
        print(f"{now.split('.')[:-1]}: {s}")

def recursive_print(m, level):
    if type(m) == dict or type(m) == OrderedDict:
        for k, v in m.items():
            if level==0:
                if 'model' in k:
                    print(f'{"#", "#"*(level*4), k}', flush=True)
                    recursive_print(v, level=level+1)
            else:
                if type(v) == torch.Tensor:
                    print(f'{"#"*(level*4), k, v.shape}')
                else:
                    print(f'{"#"*(level*4), k}')
                recursive_print(v, level=level+1)

def parse_output_path(path_to_checkpoint, output_path):
    iter_id = CP_ID_PATTERN.search(path_to_checkpoint).group()
    out = output_path
    output_path = os.path.join(out, iter_id)
    return output_path

def add_or_combine_to_dict(target, shard, target_key, dim=0):
    target_value = target.get(target_key)
    # key = new_key if new_key else target_key
    if target_value != None:
        target[target_key] = torch.cat([target_value, shard], dim=dim)
        log(f"Adding {target_key}. New shape: {target[target_key].shape}", mode='debug')
    else:
        target[target_key] = shard

# 
def combine_swiglu_mlp(encoder):
    up_layer_keys = sorted([k for k in encoder.keys() if "h_to_4h.weight.up_proj" in k])
    gate_layer_keys = sorted([k for k in encoder.keys() if "h_to_4h.weight.gate_proj" in k])

    for (up_key, gate_key) in zip(up_layer_keys, gate_layer_keys):
        up = encoder.pop(up_key)
        gate = encoder.pop(gate_key)
        # delete temp proj keys
        encoder[".".join(up_key.split(".")[:-1])] = torch.cat([up, gate], dim=0)




def merge_model(path_to_checkpoint, mode='default'):

    chunks = [pt.absolute() for pt in Path(path_to_checkpoint).glob("*/model_optim_rng.pt")]
    chunks = sorted(chunks)
    
    log("Found chunks")
    for chunk in chunks:
        print(f"## {chunk}")
    print()
    cp_args = None 
    vp_size = 0
    tp_size = 0
    pp_size = 0
    vp_layers = 0
    encoder = {}
    word_embeddings = {}
    output_layer = {}
    iteration = ""
    cp_version = ""
    tokens = ""
    group_query_key = ""
    # Looping order: TP_PP 00_000, 00_001, ..., 01_000, ... 
    for i, chunk in enumerate(chunks):

        if len(chunks) == 1:
            tp_rank = pp_rank = 0
        else:
            tp_rank, pp_rank = PARALLEL_RANK_PATTERN.search(str(chunk)).groups()
            tp_rank = int(tp_rank)
            try:
                pp_rank = int(pp_rank)
            except:
                pp_rank = 0


        log(f"processing # {chunk.absolute()}")
        log(f"tp_rank: {tp_rank}, pp_rank: {pp_rank}")

        shard = torch.load(chunk, map_location='cuda')
        if i == 0:
            cp_args = shard['args']
            vp_size = cp_args.virtual_pipeline_model_parallel_size if cp_args.virtual_pipeline_model_parallel_size else 1
            vp_layers = cp_args.num_layers_per_virtual_pipeline_stage
            pp_size = cp_args.pipeline_model_parallel_size
            tp_size = cp_args.tensor_model_parallel_size
            num_layers = cp_args.num_layers
            iteration = shard['iteration']
            cp_version = shard['checkpoint_version']
            tokens = shard.get('tokens', None)
            group_query_key = 'num_key_value_heads' if vars(cp_args).get('num_key_value_heads') else 'num_query_groups'

            assert int(pp_size)*int(tp_size)==len(chunks), "Number of shard paths and no. shards in arguments differ!"
        
        for vp_rank in range(vp_size):
            # test vp offsetting by having model splitted up into layers / (pp_size *vp_layers). With layers=40, pp=4, vp_layers=5 -> model is split into shards: 5x4x2
            if vp_size == 1:
                lm = shard['model']['language_model']
                vp_offset = 0
                pp_offset = int(pp_rank * num_layers/pp_size)
            else:
                lm = shard[f'model{vp_rank}']['language_model']
                vp_offset = vp_rank * (pp_size * vp_layers) 
                pp_offset = pp_rank * vp_layers
            log(f"model{vp_rank}, {pp_rank}, {vp_rank}", mode='debug')
            conv = lambda s: f".{str(int(s.groups()[0]) + pp_offset + vp_offset)}."

            if vp_rank == 0 and pp_rank == 0:
                # handle word embeddings / tensor parallel level
                embedding_shard = lm['embedding']['word_embeddings']['weight']
                add_or_combine_to_dict(word_embeddings, embedding_shard, target_key='weight')

            if pp_rank == (pp_size-1) and vp_rank == (vp_size-1):
                # convert Namespace-object to dict 
                if vars(cp_args).get('untie_embeddings_and_output_weights'):
                    log("using untied embeddings", mode='debug')
                    output_layer_shard = lm['output_layer']['weight']
                    add_or_combine_to_dict(output_layer, output_layer_shard, target_key='weight')
            
            for name, layer in lm['encoder'].items():
                
                layer = layer.to(DEVICE)
                layer_name = re.sub("\.(\d*)\.", conv, name)
                log(f"{name}  --->  {layer_name}", mode='debug')
                # state_dict_layer = encoder.get(layer_name)

                if cp_args.swiglu:
                    if "mlp.dense_h_to_4h" in name:
                        up_proj, gate_proj = torch.chunk(layer, 2, dim=0)
                        # print("MLP shapes:", up_proj.shape, gate_proj.shape)
                        up_proj_key = layer_name + ".up_proj"
                        gate_proj_key = layer_name + ".gate_proj"
                        add_or_combine_to_dict(encoder, up_proj, up_proj_key, dim=0)
                        add_or_combine_to_dict(encoder, gate_proj, gate_proj_key, dim=0)
                    else:
                        if ('self_attention.dense.weight' in name) or \
                            ('mlp.dense_4h_to_h' in name):
                            add_or_combine_to_dict(encoder, layer, layer_name, dim=1)
                        elif "norm" in layer_name:
                            if tp_rank == 0:
                                # only take layernorms from the first layers
                                add_or_combine_to_dict(encoder, layer, layer_name)
                        elif vars(cp_args).get(group_query_key) != cp_args.num_attention_heads \
                                and ('self_attention.query_key_value.weight' in name):
                             
                            # [sq, b, ((nq + 2 * nkv) * hn)] --> [sq, b, nkv, (nq // nkv + 2), hn] 
#                            shape = (-1,
#                            # TODO: fix to divide with num_key_value_groups
#                                cp_args.num_attention_heads // vars(cp_args).get(group_query_key) + 2, 
#                                cp_args.hidden_size // cp_args.num_attention_heads, 
#                                cp_args.hidden_size)

                            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
                                # mixed_x_layer.size()[:-1] + (
                                #     self.num_query_groups_per_partition,
                                #     (
                                #         (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                                #         * self.hidden_size_per_attention_head
                                #     ),
                                # )
                            print(vars(cp_args))
                            if cp_args.group_query_attention:
                                num_query_groups = vars(cp_args).get(group_query_key)
                                
                                parts = cp_args.num_attention_heads // vars(cp_args).get(group_query_key) + 2
                            else:
                                parts = 3
                            # print all items forming the shape
                            print(f"cp_args.num_attention_heads: {cp_args.num_attention_heads} \n" +
                                f"vars(cp_args).get(group_query_key): {vars(cp_args).get(group_query_key)} \n" +
                                f"cp_args.hidden_size: {cp_args.hidden_size} \n" +
                                f"layer.shape: {layer.shape} \n")
                            shape = (-1,
                                parts, 
                                cp_args.hidden_size // cp_args.num_attention_heads, 
                                cp_args.hidden_size)

 
                            print("shape:", shape)
                            layer = layer.view(*shape)
                            # print number of elements
                            print("Number of elements in layer:", layer.numel())


                            query_layer = layer[:,:-2].reshape(-1, cp_args.hidden_size)
                            key_value_layer = layer[:,-2:].reshape(-1,cp_args.hidden_size)
                            kv_label = ''.join(layer_name.split('query_'))
                            q_label = ''.join(layer_name.split('_key_value'))

                            add_or_combine_to_dict(encoder, query_layer, q_label, dim=0)
                            add_or_combine_to_dict(encoder, key_value_layer, kv_label, dim=0)

                        else:
                            add_or_combine_to_dict(encoder, layer, layer_name, dim=0)

                else:
                    # TODO: reformat above condition to only effect mlp_dense_h_to_4h
                    add_or_combine_to_dict(encoder, layer, layer_name)

    print("Args:",cp_args)
    # encoder['output_layer'] = output_layer 
    cp_args.pipeline_model_parallel_size = 1
    cp_args.tensor_model_parallel_size = 1
    combine_swiglu_mlp(encoder)
    # Combine into a single state_dict
    state_dict = { 
        "model": {
            "language_model": {
                "embedding" : {
                    "word_embeddings": word_embeddings
                },
                "encoder": encoder,
                "output_layer": output_layer

            }
        },
        "args": cp_args,
        "iteration": iteration,
        "checkpoint_version": cp_version,
        "tokens": tokens
    }
    return state_dict


def save_model(state_dict, output_path):
    if not os.path.exists(os.path.join(output_path, "mp_rank_00")):
        os.makedirs(os.path.join(output_path, "mp_rank_00"))
    
    # Save latest iteration for megatron loader
    iter_path = os.path.join('/'.join(output_path.split("/")[:-1]), 'latest_checkpointed_iteration.txt')
    with open(iter_path, 'w') as c_out:
        c_out.write(str(state_dict["iteration"]))
    parsed_output_path = os.path.join(output_path, "mp_rank_00", "model_optim_rng.pt") 
    torch.save(state_dict, parsed_output_path)
    log(    f"Succesfully saved the model to {parsed_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge sharded Megatron-model into a single modelfile")
    parser.add_argument(
        "--path_to_unmerged_checkpoint",
        type=str,
        help="Path to the checkpoint file .pt file",
    )
    parser.add_argument(
        "--output_path",
        help='Path to the output directory to store the converted checkpoint',
        required=True,
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='More detailed print about progress'
    )

    args = parser.parse_args()
    if args.debug:
        global CUR_MODE
        CUR_MODE = 'debug'
    
    output_path = parse_output_path(args.path_to_unmerged_checkpoint, args.output_path)
    if os.path.exists(os.path.join(output_path, "mp_rank_00")):
        assert False, "Output path already exists and should be empty!"
    log("Starting merging")
    state_dict = merge_model(args.path_to_unmerged_checkpoint)
    log("...Done")
    log("Starting saving the model")
    save_model(state_dict, output_path)
    log("...Done")


if __name__ == '__main__':
    main()
