import torch

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def unet_add_coded_conds(unet, added_number_count=1):
    unet.add_time_proj = Timesteps(256, True, 0)
    unet.add_embedding = TimestepEmbedding(256 * added_number_count, 1280)

    def get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs):
        coded_conds = added_cond_kwargs.get("coded_conds")
        batch_size = coded_conds.shape[0]
        time_embeds = unet.add_time_proj(coded_conds.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb)
        aug_emb = unet.add_embedding(time_embeds)
        return aug_emb

    unet.get_aug_embed = get_aug_embed

    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        cross_attention_kwargs = {k: v for k, v in kwargs['cross_attention_kwargs'].items()}
        coded_conds = cross_attention_kwargs.pop('coded_conds')
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs

        coded_conds = torch.cat([coded_conds] * (sample.shape[0] // coded_conds.shape[0]), dim=0).to(sample.device)
        kwargs['added_cond_kwargs'] = dict(coded_conds=coded_conds)
        return unet_original_forward(sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_unet_forward

    return
