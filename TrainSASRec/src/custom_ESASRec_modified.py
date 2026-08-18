"""
SASRec (copied from models.py for customization).
"""

import numpy as np
import pickle
import warnings

import torch
from torch import nn

from models import PointWiseFeedForward

class ESASRecV2(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=128, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, path_to_cont_embeddings=None, content_embeddings_dim=2048, adapter_dropout_rate=0, mode_1 = "only_adapter", mode_2 = "only_adapter",
                 after_sum_norm = None,
                 scale = False
                 
                 ):
    
        super(ESASRecV2, self).__init__()
        if mode_1 == "only_adapter":
            self.mode_1 = "only_adapter"
        elif mode_1 == "only_item_emb":
            self.mode_1 = "only_item_emb"
        elif mode_1 == "adapter_and_item_emb":
            self.mode_1 = "adapter_and_item_emb"
        else:
            raise ValueError(f"Invalid mode_1: {mode_1}")

        if mode_2 == "only_adapter":
            self.mode_2 = "only_adapter"
        elif mode_2 == "only_item_emb":
            self.mode_2 = "only_item_emb"
        elif mode_2 == "adapter_and_item_emb":
            self.mode_2 = "adapter_and_item_emb"
        else:
            raise ValueError(f"Invalid mode_2: {mode_2}")

        if after_sum_norm is None:
            self.after_sum_norm = nn.Identity()
        elif after_sum_norm == "layer_norm":
            self.after_sum_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        elif after_sum_norm == "batch_norm":
            self.after_sum_norm = nn.BatchNorm1d(hidden_units)
        elif after_sum_norm == "rms_norm":
            self.after_sum_norm = nn.RMSNorm(hidden_units)
        else:
            raise ValueError(f"Invalid after_sum_norm: {after_sum_norm}")
        self.scale = scale

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.content_embeddings_dim = content_embeddings_dim

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        # Без файла — как item_emb: N(0, initializer_range²), padding 0; из файла — в load_embeddings.
        # load_embeddings вызывается после apply(_init_weights): иначе _init_weights перезапишет
        # веса, загруженные из pickle (см. nn.Embedding в _init_weights).
        self.item_content_emb = nn.Embedding(item_num + 1, content_embeddings_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        bottleneck = content_embeddings_dim // 2
        self.adapter = nn.Sequential(
            nn.Linear(content_embeddings_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_units),
            nn.Dropout(adapter_dropout_rate),
            nn.LayerNorm(hidden_units, eps=1e-8),
        )
        self.stage = 0
        #self.after_sum_layer_norm = after_sum_layer_norm

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units,
                                                   num_heads,
                                                   dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)
        self.load_embeddings(path_to_cont_embeddings)

    def load_embeddings(self, path_to_embeddings):
        
        # None: не подгружаем файл — item_content_emb остаётся как после __init__ (та же случайная инициализация, что у item_emb).
        if path_to_embeddings is None:
            return

        with open(path_to_embeddings, "rb") as f:
            mapping_id_embeddings = pickle.load(f)

        n = len(mapping_id_embeddings)
        # ключи 0..n-1 подряд — собираем строки в порядке id
        rows = [mapping_id_embeddings[i] for i in range(n)]
        emb = torch.tensor(rows, dtype=torch.float32)
        if emb.shape[0] != self.item_num + 1:
            warnings.warn(
                f"item_content_emb rows: expected item_num+1={self.item_num + 1}, got {emb.shape[0]}",
                stacklevel=2,
            )
        if emb.shape[1] != self.content_embeddings_dim:
            warnings.warn(
                f"item_content_emb dim: expected content_embeddings_dim={self.content_embeddings_dim}, got {emb.shape[1]}",
                stacklevel=2,
            )


        # # Случайная перестановка строк (id смотрит на чужой вектор), строка 0 (padding) без изменений
        # n_rows = emb.size(0)
        # perm = torch.randperm(n_rows - 1)
        # emb = torch.cat([emb[:1], emb[1:n_rows][perm]], dim=0)

        # print("EMB 0",emb[0])
        # print("EMB 2",emb[2])
        # print("EMB 10",emb[10])
        self.item_content_emb = nn.Embedding.from_pretrained(emb, freeze=True, padding_idx=0)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _embed_input_sequence(self, input_ids):
        if self.mode_1 == "only_adapter" and self.stage == 2:
            sum = self.adapter(self.item_content_emb(input_ids))
        elif self.mode_1 == "only_item_emb" and self.stage == 2:
            sum = self.item_emb(input_ids)
        elif self.mode_1 == "adapter_and_item_emb":
            sum = self.adapter(self.item_content_emb(input_ids)) + self.item_emb(input_ids)
        else:
            raise ValueError(f"Invalid mode_1: {self.mode_1} or stage: {self.stage}")
        return self.after_sum_norm(sum)

    def _make_new_item_emb_weight(self):

        if self.mode_2 == "only_adapter" and self.stage == 2:
            sum = self.adapter(self.item_content_emb.weight)
        elif self.mode_2 == "only_item_emb" and self.stage == 2:
            sum = self.item_emb.weight
        elif self.mode_2 == "adapter_and_item_emb":
            sum = self.adapter(self.item_content_emb.weight) + self.item_emb.weight
        else:
            raise ValueError(f"Invalid mode_2: {self.mode_2} or stage: {self.stage}")

        return self.after_sum_norm(sum)


    def content_hidden_state(self, input_ids):
        if self.stage == 0:
            raise ValueError("content_hidden_state is not available in stage 0")
        return self.adapter(self.item_content_emb(input_ids))

    # parameter attention mask added for compatibility with Lightning module, not used
    def forward(self, input_ids, attention_mask):

        seqs = self._embed_input_sequence(input_ids)
        #seqs = self.after_sum_layer_norm(seqs)
        if self.scale:
            seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(outputs, self._make_new_item_emb_weight().transpose(0, 1))

        return outputs

    def get_embeddings(self, input_ids):
        if self.training:
            raise RuntimeError(
                "get_embeddings() is only valid in eval mode; call model.eval() before using it."
            )
        return self.adapter(self.item_content_emb(input_ids))


