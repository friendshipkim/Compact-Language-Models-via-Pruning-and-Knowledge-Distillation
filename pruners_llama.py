import torch.nn as nn
import torch
from models import Block
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def prune_linear(module, idx, axis="out") -> None:
    # Linear(in_features, out_features): weight shape (out_features, in_features)
    if isinstance(module, nn.Linear):
        if axis == "out":
            module.out_features = idx.size(0)
            # check if already pruned
            if module.weight.data.shape[0] == idx.size(0):
                return
            module.weight.data = module.weight.data[idx, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[idx]
        elif axis == "in":
            module.in_features = idx.size(0)
            # check if already pruned
            if module.weight.data.shape[1] == idx.size(0):
                return
            module.weight.data = module.weight.data[:, idx]


def prune_layernorm(module, idx) -> None:
    if isinstance(module, LlamaRMSNorm) or isinstance(module, nn.LayerNorm):
        module.weight.data = module.weight.data[idx]
        module.normalized_shape = idx.size(0)


def prune_embedding(module, idx) -> None:
    if isinstance(module, nn.Embedding):
        module.weight.data = module.weight.data[:, idx]
    # change the embedding size
    module.embedding_dim = idx.size(0)


def prune_intermediate(model, ratio=0.2, batch_agg="l2", is_random=False) -> None:
    # goal: trim the MLP layer weights
    intermediate_size = model.config.intermediate_size
    intermediate_size_pruned = int((1 - ratio) * intermediate_size)
    
    idx_list = []
    for layer in model.model.layers:
        # fetch the importances
        importances = layer.mlp.down_proj.calculated_importance
    
        # aggregate batch dimension
        if batch_agg == "l1":
            importances = importances.norm(dim=0, p=1)
        elif batch_agg == "l2":
            importances = importances.norm(dim=0, p=2)
        else:
            raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
        
        idx = importances.argsort(descending=True)[:intermediate_size_pruned]
        
        if is_random:
            idx = torch.randperm(intermediate_size)[:intermediate_size_pruned]
        
        idx_list.append(idx)
        
        # prune weights
        prune_linear(layer.mlp.gate_proj, idx, axis="out")
        prune_linear(layer.mlp.up_proj, idx, axis="out")
        prune_linear(layer.mlp.down_proj, idx, axis="in")
    print("MLP pruned!")
    model.config.intermediate_size = intermediate_size_pruned
    return model, idx_list


def prune_attn_heads(model, ratio=0.2, batch_agg="l2", is_random=False) -> None:
    if ratio == 0.0:
        return model, None
    
    for layer in model.model.layers:
        importances = layer.self_attn.o_proj.calculated_importance
        
        # aggregate batch dimension
        if batch_agg == "l1":
            importances = importances.norm(dim=0, p=1)
        elif batch_agg == "l2":
            importances = importances.norm(dim=0, p=2)
        else:
            raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
        
        # TODO prune heads
            
            
def prune_hidden(model, ratio=0.2, batch_agg="l2", is_random=False) -> None:
    # goal: trim the hidden dimension of the weight matrices in MLP, MHA, and LayerNorm layers.
    hidden_size = model.config.hidden_size
    hidden_size_pruned = int((1 - ratio) * hidden_size)
    
    # fetch the importances across all layers
    importances = [
        abs(layer.input_layernorm.calculated_importance) + abs(layer.post_attention_layernorm.calculated_importance)
        for layer in model.model.layers
    ]
    importances = torch.stack(importances, dim=0)
    # reduce layer dimension
    importances = importances.sum(dim=0)
    
    # reduce batch dimension
    if batch_agg == "l1":
        importances = importances.norm(dim=0, p=1)
    elif batch_agg == "l2":
        importances = importances.norm(dim=0, p=2)
    else:
        raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
    
    idx = importances.argsort(descending=True)[:hidden_size_pruned]
    
    if is_random:
        idx = torch.randperm(hidden_size)[:hidden_size_pruned]
    
    # now let's prune the model
    # embedding layer
    prune_embedding(model.model.embed_tokens, idx)
    for layer in model.model.layers:
        prune_layernorm(layer.input_layernorm, idx)
        prune_linear(layer.self_attn.q_proj, idx, axis="in")
        prune_linear(layer.self_attn.k_proj, idx, axis="in")
        prune_linear(layer.self_attn.v_proj, idx, axis="in")
        prune_linear(layer.self_attn.o_proj, idx, axis="out")
        
        prune_linear(layer.mlp.gate_proj, idx, axis="in")
        prune_linear(layer.mlp.up_proj, idx, axis="in")
        prune_linear(layer.mlp.down_proj, idx, axis="out")
        prune_layernorm(layer.post_attention_layernorm, idx)
    
    # ln
    prune_layernorm(model.model.norm, idx)
        
    # lm head is tied to the embedding layer
    prune_linear(model.lm_head, idx, axis="in")
    assert torch.allclose(model.lm_head.weight.data, model.model.embed_tokens.weight.data)
    
    model.config.hidden_size = hidden_size_pruned
    print("Embeddings pruned!")
    return model, idx


AVAILABLE_PRUNING_STRATEGIES = {
    "width_attn": prune_attn_heads,
    "width_hidden": prune_hidden,
    "width_intermediate": prune_intermediate,
}
