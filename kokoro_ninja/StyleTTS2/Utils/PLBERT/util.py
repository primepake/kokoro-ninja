import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu', weights_only=False)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    del new_state_dict["embeddings.position_ids"]

    # Vietnamese vocab extension: if the model was built with a larger vocab
    # (e.g. 187) than the pretrained PL-BERT (178), warm-start by copying the
    # first N pretrained rows of the word_embeddings into the larger tensor and
    # keeping the (already initialized) new rows. Avoids a size-mismatch error
    # and preserves pretrained phoneme embeddings for indices 0..N-1.
    we_key = "embeddings.word_embeddings.weight"
    if we_key in new_state_dict:
        ckpt_we = new_state_dict[we_key]
        model_we = bert.state_dict()[we_key]
        if ckpt_we.shape != model_we.shape:
            assert ckpt_we.shape[1] == model_we.shape[1] and model_we.shape[0] >= ckpt_we.shape[0], \
                f"Incompatible PL-BERT word_embeddings: ckpt {tuple(ckpt_we.shape)} vs model {tuple(model_we.shape)}"
            merged = model_we.clone()
            merged[:ckpt_we.shape[0]] = ckpt_we
            new_state_dict[we_key] = merged
            print("load_plbert: warm-started word_embeddings %s -> %s (copied first %d rows)"
                  % (tuple(ckpt_we.shape), tuple(model_we.shape), ckpt_we.shape[0]))

    bert.load_state_dict(new_state_dict, strict=False)

    return bert
