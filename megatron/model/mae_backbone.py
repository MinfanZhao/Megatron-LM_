"""Masked Autoencoder Vision Transformer(MAEVIT) model."""

import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron import get_args
from megatron.model.mae_transformer import ParallelTransformer
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from megatron.model.module import MegatronModule
from megatron.model.enums import LayerType

CLASS_TOKEN_LENGTH = 1
class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = args.hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert(isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = args.class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class MaskedAutoencoderViT(MegatronModule):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(self,
                 pre_process=True,
                 post_process=True,
                 class_token=True,
                 single_token_output=False,
                 post_layer_norm=True,
                 drop_path_rate=0.0,
                 add_encoder=True,
                 add_decoder=True):
        super(MaskedAutoencoderViT, self).__init__(share_word_embeddings=False)
        args = get_args()

        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, args.num_layers
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.post_layer_norm = post_layer_norm
        self.hidden_size = args.hidden_size
        self.decoder_embed_dim = args.decoder_embed_dim # add to args
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        self.in_chans = args.num_channels
        self.mask_ratio = args.mask_ratio # add to args
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (1 if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None

        if self.add_encoder:
            if self.pre_process:
                # cls_token
                if self.class_token:
                    self.cls_token = torch.nn.Parameter(
                        torch.randn(1, 1, self.hidden_size)
                    )
                    torch.nn.init.zeros_(self.cls_token)
                self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
                
                # Linear encoder
                self.linear_encoder = torch.nn.Linear(
                    self.flatten_dim, self.hidden_size
                )

                # embedding
                self.position_embeddings = torch.nn.Embedding(
                    self.seq_length, self.hidden_size
                )
                init_method_normal(args.init_method_std)(
                    self.position_embeddings.weight
                )

                args.class_token_present = self.class_token
                self.position_embeddings._register_load_state_dict_pre_hook(
                    twod_interpolate_position_embeddings_hook
                )

                self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)
        
        ## encoder
            self.encoder = ParallelTransformer(
                self.init_method,
                self.scaled_init_method,
                pre_process=self.pre_process,
                post_process=self.post_process,
                post_layer_norm=self.post_layer_norm,
                drop_path_rate=self.drop_path_rate
            )
            self._encoder_key = 'encoder'
        else:
            self.encode = None
        
        ## decoder 
        if self.add_decoder:
            if self.pre_process:
                self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
                self.decoder_embed = torch.nn.Linear(self.hidden_size, self.decoder_embed_dim, bias=True)
                self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
                self.decoder_pos_embed = torch.nn.Embedding(self.seq_length, self.decoder_embed_dim) 

            self.decoder = ParallelTransformer(
                self.init_method,
                self.scaled_init_method,
                layer_type=LayerType.mae_decoder,
                pre_process=self.pre_process,
                post_process=self.post_process,
                post_layer_norm=self.post_layer_norm,
                drop_path_rate=self.drop_path_rate
            )
            self._decoder_key = 'mae_decoder'
            if self.post_process:
                self.decoder_pred = torch.nn.Linear(self.decoder_embed_dim, self.flatten_dim, bias=True)
        else:
            self.decoder = None

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence, # batch, length, dim
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove, 
        ids_restore = torch.argsort(ids_shuffle, dim=1) # ascend

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if self.add_encoder and self.pre_process:
            pass
        elif self.add_encoder:
            assert len(input_tensor) == 3, \
                'input_tensor should be length 3 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
            self.ids_restore = input_tensor[1]
            self.mask = input_tensor[2]
        elif self.add_decoder and self.pre_process:
            assert len(input_tensor) == 3, \
                'input_tensor should be length 3 for stage-1 with only decoder'
            self.encoder_output = input_tensor[0]
            self.ids_restore = input_tensor[1].long()
            self.mask = input_tensor[2]
        elif self.add_decoder:
            assert len(input_tensor) == 2, \
                'input_tensor should be length 2 for stage with only decoder'
            self.decoder.set_input_tensor(input_tensor[0])
            self.mask = input_tensor[1]
        else:
            raise Exception('Stage must have at least either encoder or decoder')



    def forward(self, input):

        if self.pre_process and self.add_encoder:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half

            #patch embedding
            encoder_output = self.linear_encoder(rearranged_input) # b,s,h

            # add pos embed w/o cls token
            pos_embedded = encoder_output + \
                    self.position_embeddings(self.position_ids[:, 1:])
            
            # masking : s -> s * (1-mask_ratio)
            masked_output, mask, ids_restore = self.random_masking(pos_embedded, self.mask_ratio)
            concatenated_tokens = masked_output

            # append cls token
            if self.class_token:
                cls_tokens = self.cls_token.expand(masked_output.shape[0], -1, -1) # b, CLASS_TOKEN_LENGTH, hidden_size
                concatenated_tokens = torch.cat((cls_tokens, masked_output), dim=1)

            # [b, s, h] => [s, b, h]
            # token_embeddings = concatenated_tokens.transpose(0, 1).contiguous()
            # hidden_states = self.embedding_dropout(token_embeddings)
            hidden_states = concatenated_tokens.transpose(0, 1).contiguous()
            self.ids_restore = ids_restore
            self.mask = mask

        else:
            hidden_states = None


        if self.add_encoder:
            hidden_states = self.encoder(hidden_states, None)
            # size: [s*(1-mask_ratio) b h] [b s] [b s]
            return hidden_states, self.ids_restore, self.mask

        # decoder forward
        # if is the first stage of decoder, input should come from encoder_output
        if self.add_decoder:
            if self.pre_process:
                encoder_input = self.encoder_output #[s b h]
                hidden_states = encoder_input.transpose(0,1).contiguous() # [b s h]
                # embed tokens
                x = self.decoder_embed(hidden_states)

                # append mask tokens to sequence
                
                mask_tokens = self.mask_token.repeat(x.shape[0], self.ids_restore.shape[1] + 1 - x.shape[1], 1)
                x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
                x_ = torch.gather(x_, dim=1, index=self.ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

                x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

                # add pos embed
                hidden_states = x + self.decoder_pos_embed(self.position_ids[:, :])
                hidden_states = hidden_states.transpose(0,1).contiguous() # [s b h]

        
            hidden_states = self.decoder(hidden_states, None)

            if self.post_process :
                hidden_states = self.decoder_pred(hidden_states)
                hidden_states = hidden_states[1:, :, :] # remove token
                # # [s b h] => [b s h]
                # if self.single_token_output:
                #     hidden_states = hidden_states[0]
                # else:
                    
                hidden_states = hidden_states.transpose(0, 1).contiguous()
                
            return hidden_states, self.mask


