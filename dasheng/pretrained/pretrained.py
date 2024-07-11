import torch
from einops import rearrange
from ..train.models import AudioTransformerMAE_Encoder

PRETRAINED_CHECKPOINTS = {
    'dasheng_base':
    'https://zenodo.org/records/11511780/files/dasheng_base.pt?download=1',
    'dasheng_06B':
    'https://zenodo.org/records/11511780/files/dasheng_06b.pt?download=1',
    'dasheng_12B':
    'https://zenodo.org/records/11511780/files/dasheng_12b.pt?download=1',
}


#Using the pretrained encoders, remove all masking
class Dasheng(AudioTransformerMAE_Encoder):

    # need the *args, **kwargs otherwise we get a linter warning
    def forward_features(self, x: torch.Tensor, *args,
                         **kwargs) -> torch.Tensor:
        x = self.patch_embed(x)
        *_, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.target_length:
            splits = x.split(self.target_length, -1)

            padding_start_in_chunks = 0
            if len(splits) > 1 and (splits[-1].shape[-1] < self.target_length):
                pad = torch.zeros(*x.shape[:-1],
                                  self.target_length,
                                  device=x.device)
                pad[..., :splits[-1].shape[-1]] = splits[-1]
                padding_start_in_chunks = splits[-1].shape[-1] // self.patch_stride[-1]
                splits = torch.stack((*splits[:-1], pad), dim=0)
            n_splits = len(splits)
            # Splits into the batch_size, speeds up computation
            x = rearrange(splits, 'spl b c f t-> (spl b) c f t')
            x = self.forward_features(x)
            x = rearrange(x, '(spl b) ... d -> spl b (...) d', spl=n_splits)
            # Trim last chunk
            if padding_start_in_chunks > 0:
                not_padded = rearrange(x[:-1], 'spl b ... d -> b (spl ...) d')
                padded = x[-1][:, :padding_start_in_chunks, ...]
                x = torch.cat((not_padded, padded), dim=1)
            else:
                x = rearrange(x, 'spl b ... d -> b (spl ...) d', spl=n_splits)
            return x
        else:
            return self.forward_features(x)



    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        x = self.forward_to_spec(x)
        return self.forward_spectrogram(x)

    @classmethod
    def from_pretrained(
            cls, pretrained_url: str,
            **additional_model_kwargs) -> AudioTransformerMAE_Encoder:
        """
        Class method to create a new Dasheng model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        if 'http' in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(pretrained_url,
                                                      map_location='cpu')
        else:
            dump = torch.load(pretrained_url, map_location='cpu')
        model_parmeters, model_config = dump['model'], dump['config']
        instance = cls(**{**model_config, **additional_model_kwargs})
        instance.load_state_dict(model_parmeters, strict=True)
        return instance


def dasheng_base(**model_kwargs):
    model_kwargs["embed_dim"] = 768
    model_kwargs["depth"] = 12
    model_kwargs["num_heads"] = 12
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_base'],
                                   **model_kwargs)


def dasheng_06B(**model_kwargs):
    model_kwargs["embed_dim"] = 1280
    model_kwargs["depth"] = 32
    model_kwargs["num_heads"] = 16
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_06B'],
                                   **model_kwargs)


def dasheng_12B(**model_kwargs):
    model_kwargs["embed_dim"] = 1536
    model_kwargs["depth"] = 40
    model_kwargs["num_heads"] = 24
    return Dasheng.from_pretrained(PRETRAINED_CHECKPOINTS['dasheng_12B'],
                                   **model_kwargs)


if __name__ == "__main__":
    mdl = dasheng_base()
    print(mdl(torch.randn(1, 168499)).shape)
