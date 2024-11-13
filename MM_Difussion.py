import torch.nn as nn
import torch
from shap_e.models.generation.transformer import PointDiffusionTransformer
from shap_e.models.generation.pretrained_clip import ImageCLIP, FrozenImageCLIP, ImageType
from shap_e.models.generation.util import timestep_embedding
import math
from typing import Optional, Iterable, Union, Dict, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================= Early Fusion Mechanisms =================================

# Fusion mode 100's
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


# Fusion mode 200's
class RowWiseGatedFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(RowWiseGatedFusionModule, self).__init__()
        # Gating mechanism for interaction
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, img_features, text_features):
        # img_features: [batch_size, 1024, 256]
        # text_features: [batch_size, 1024]
        
        # Apply gate to text features
        gated_text = self.gate(text_features)  # [batch_size, 1024]
        
        # Element-wise multiplication of gated text with each row of img_features
        fused_output = img_features * gated_text.unsqueeze(1)  # Broadcast along 1024 rows
        
        return fused_output


# Fusion mode 300's
class AttentionTextGuidedFusion(nn.Module):
    def __init__(self, img_feature_dim):
        super(AttentionTextGuidedFusion, self).__init__()
        
        # Linear layers to transform image and text features
        self.query = nn.Linear(img_feature_dim, img_feature_dim)
        self.key = nn.Linear(img_feature_dim, img_feature_dim)
        self.value = nn.Linear(img_feature_dim, img_feature_dim)
        
        
        # Multihead attention with batch_first=True
        self.multihead_attn = nn.MultiheadAttention(embed_dim=img_feature_dim, num_heads=1, batch_first=True)
        
        # Learnable scaling factor for controlling the impact of attention
        self.attn_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # Initialize weights to zero for query, key, and value transforms
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.value.weight)

    def forward(self, img_features, text_features):
        # img_features: [batch_size, 1024, img_feature_dim]
        # text_features: [batch_size, 1024, img_feature_dim]

        # Transform the image and text features into query, key, and value
        query = self.query(img_features)
        key = self.key(text_features)
        key = key.unsqueeze(1)
        value = self.value(text_features)
        value = value.unsqueeze(1)

        # Apply multi-head attention: text features attend to the image features
        attn_output, _ = self.multihead_attn(query, key, value)

        # Use the scaling factor to modulate the impact of attention
        attn_output = self.attn_scale * attn_output

        # Combine the scaled attended features with the original image features
        fused_output = img_features + attn_output

        return fused_output


# ================================= Multimodal Model =================================

class combinedCLIP(PointDiffusionTransformer):
    def __init__(
            self,
            *,
            device: torch.device,
            dtype: torch.dtype,
            fusion_mode: int = 200,
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
        self.fusion_mode = fusion_mode
        self.n_ctx = n_ctx
        self.clip = clip
        #print(self.backbone.width)
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )

        self.clip_embed_img = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )

        if self.fusion_mode < 200:
            self.cross_modal = CrossModalAttention(1024).to(device)
        elif self.fusion_mode < 300:
            self.gated_fusion = RowWiseGatedFusionModule(1024).to(device)
        elif self.fusion_mode < 400:
            self.attention_text_guided_fusion = AttentionTextGuidedFusion(img_feature_dim=1024).to(device)

        self.cond_drop_prob = cond_drop_prob


    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        # print(model_kwargs)
        model_kwargs_txt = {"texts": model_kwargs["texts"]}
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs_txt),
                        embeddings_img=self.clip.embed_images_grid(model_kwargs["images"]))



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
        # print("clip_embed")
        # print(clip_embed.shape)
        # print(clip_embed_img.shape)
        # print("t_embed")
        # print(t_embed.shape)
        # print(self.time_token_cond)
        # print("clip_embed")


        if self.fusion_mode < 200:
            clip_embed = self.cross_modal(clip_embed_img, clip_embed)
        elif self.fusion_mode < 300:
            clip_embed = self.gated_fusion(clip_embed_img, clip_embed)
        elif self.fusion_mode < 400:
            clip_embed = self.attention_text_guided_fusion(clip_embed_img, clip_embed)

        # clip_embed.unsqueeze_(1)
        # print(clip_embed_img.shape)
        # print(clip_embed.shape)
        # clip_embed = (clip_embed_img + clip_embed) / 2
        # print(clip_embed.shape)
        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]

        return self._forward_with_cond(x, cond)
