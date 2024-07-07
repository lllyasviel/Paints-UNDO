import torch


def unet_add_concat_conds(unet, new_channels=4):
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(4 + new_channels, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        cross_attention_kwargs = {k: v for k, v in kwargs['cross_attention_kwargs'].items()}
        c_concat = cross_attention_kwargs.pop('concat_conds')
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs

        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0).to(sample)
        new_sample = torch.cat([sample, c_concat], dim=1)
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_unet_forward
    return
