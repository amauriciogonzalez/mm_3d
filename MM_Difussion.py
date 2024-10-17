import torch.nn as nn
import torch
from shap_e.models.generation.transformer import PointDiffusionTransformer
from shap_e.models.generation.pretrained_clip import ImageCLIP, FrozenImageCLIP, ImageType
from shap_e.models.generation.util import timestep_embedding
import math
from typing import Optional, Iterable, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(1024, 1024)
        self.key = nn.Linear(1024, 1024)
        self.value = nn.Linear(1024, 1024)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=1, batch_first=True)
        self.expand_dim = nn.Linear(1024, 1024 * 1024)

    def forward(self, tensorI, tensorT):
        query = self.query(tensorI)
        key = self.key(tensorT)
        value = self.value(tensorT)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        output, _ = self.multihead_attn(query, key, value)
        output = output
        return output


cross_modal = CrossModalAttention(1024).to(device)

class combinedCLIP(PointDiffusionTransformer):

    def __init__(
            self,
            *,
            device: torch.device,
            dtype: torch.dtype,
            n_ctx: int = 1024,
            cond_drop_prob: float = 0.0,
            token_cond: bool = False,
            frozen_clip: bool = True,
            **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device)

        super().__init__(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + clip.grid_size ** 2,
            pos_emb_n_ctx=n_ctx,
            **kwargs,
        )
        self.device = device
        self.n_ctx = n_ctx
        self.clip = clip

        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )

        self.clip_embed_img = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )

        self.cond_drop_prob = cond_drop_prob

        # if self.text:
        #     def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        #         with torch.no_grad():
        #             return dict(embeddings=self.clip(batch_size, **model_kwargs))
        #
        # else:
        #     def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        #         _ = batch_size
        #         with torch.no_grad():
        #             return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))
        #

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            images: Optional[Iterable[Optional[ImageType]]] = None,
            texts: Optional[Iterable[Optional[str]]] = None,
            embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
            embeddings_img: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ):

        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert images is not None or embeddings is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
        if images is not None:
            clip_out_img = self.clip.embed_images_grid(images)
        else:
            clip_out_img = embeddings_img
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out_img = clip_out_img * mask[:, None, None].to(clip_out)
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        clip_out = math.sqrt(clip_out.shape[1]) * clip_out
        clip_out_img = clip_out_img.permute(0, 2, 1)  # NCL -> NLC

        clip_embed_img = self.clip_embed_img(clip_out_img)
        clip_embed = self.clip_embed(clip_out)

        clip_embed = cross_modal(clip_embed_img, clip_embed)

        print(clip_embed.shape)
        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]

        return self._forward_with_cond(x, cond)
