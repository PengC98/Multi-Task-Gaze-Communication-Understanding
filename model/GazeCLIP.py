from typing import Tuple
import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn as nn

from .CLIP_vision_encoder import CLIPVisionEncoder
from .CLIP_text_encoder import CLIPTextEncoder, TextPromptLearner
import model.resnet_scene as resnet_scene


class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)
class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""

	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)

class Attention(nn.Module):
	'''
	A generalized attention module with more flexibility.
	'''

	def __init__(
			self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
			qk_proj_dim: int, v_proj_dim: int, num_heads: int,
			out_dim: int
	):
		super().__init__()

		self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
		self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
		self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
		self.out_proj = nn.Linear(v_proj_dim, out_dim)

		self.num_heads = num_heads
		assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

		self._initialize_weights()

	def _initialize_weights(self):
		for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
			nn.init.xavier_uniform_(m.weight)
			nn.init.constant_(m.bias, 0.)

	def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):
		assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
		N = q.size(0);
		assert k.size(0) == N and v.size(0) == N
		Lq, Lkv = q.size(1), k.size(1);
		assert v.size(1) == Lkv

		q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

		H = self.num_heads
		Cqk, Cv = q.size(-1) // H, v.size(-1) // H

		q = q.view(N, Lq, H, Cqk)
		k = k.view(N, Lkv, H, Cqk)
		v = v.view(N, Lkv, H, Cv)


		aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)

		if mask is not None:
			mask_value = -torch.finfo(aff.dtype).max

			aff.masked_fill_(~mask, mask_value)

		aff = aff.softmax(dim=-2)
		mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)


		out = self.out_proj(mix.flatten(-2))



		return out

class ResAttblock(nn.Module):
	def __init__(self, in_dim: int, num_heads: int,
			out_dim: int):
		super().__init__()
		self.attn = Attention(
			q_in_dim=in_dim, k_in_dim=in_dim, v_in_dim=in_dim,
			qk_proj_dim=in_dim, v_proj_dim=in_dim, num_heads=num_heads, out_dim=out_dim
		)
		self.ln_1 = nn.LayerNorm(out_dim)
		self.mlp =  nn.Sequential(
			nn.Linear(out_dim,2*out_dim),
			QuickGELU(),
			nn.Linear(out_dim*2,out_dim),
		)
		self.ln_2 = nn.LayerNorm(out_dim)
	def forward(self,x,k,v,mask=None):
		x = x + self.attn(self.ln_1(x), self.ln_1(k),self.ln_1(v),mask)
		x = x + self.mlp(self.ln_2(x))
		return x




class ITTrans(nn.Module):
	def __init__(self, in_dim: int,out_dim: int,T,layers=1):
		super().__init__()
		transformer_heads = in_dim // 64
		self.resblocks = ResAttblock(in_dim, transformer_heads,out_dim)

	def forward(self,x,k,v):
		ori_x = x

		x = self.resblocks(x,k,v)

		x = x.type(ori_x.dtype) + ori_x
		return x

class GazeCLIP(nn.Module):

	def __init__(
			self,
			device,
			# load weights
			backbone_path: str = '',
			# data shape
			input_size: Tuple[int, int] = (224, 224),
			num_frames: int = 20,
			# model def
			feature_dim: int = 768,
			patch_size: Tuple[int, int] = (16, 16),
			num_heads: int = 12,
			num_layers: int = 12,
			mlp_factor: float = 4.0,
			embed_dim: int = 512,

			text_context_length: int = 77,
			text_vocab_size: int = 49408,
			text_transformer_width: int = 512,
			text_transformer_heads: int = 8,
			text_transformer_layers: int = 12,
			text_num_prompts: int = 8,
			text_prompt_pos: str = 'end',
			text_prompt_init: str = '',
			text_prompt_CSC: bool = False,
			text_prompt_classes_path: str = '',
			text_prompt_classes_path_atomic: str = '',
			# zeroshot eval
			zeroshot_evaluation: bool = False,
			zeroshot_text_features_path: str = '',
	):
		super().__init__()

		# frames and tubelet
		self.num_frames = num_frames

		# use summary token


		# clip loss logit_scale
		self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		# zeroshot text_features
		self.zeroshot_evaluation = zeroshot_evaluation
		if self.zeroshot_evaluation:
			self.text_features = torch.load(zeroshot_text_features_path, map_location='cpu')

		# visual model
		self.visual = CLIPVisionEncoder(
			device,
			# data shape
			input_size=input_size,
			num_frames=num_frames,
			# model def
			feature_dim=feature_dim,
			patch_size=patch_size,
			num_heads=num_heads,
			num_layers=num_layers,
			mlp_factor=mlp_factor,
			embed_dim=embed_dim,
			# use summary token
		)

		# text prompt learning

		self.textual = CLIPTextEncoder(
			embed_dim=embed_dim,
			context_length=text_context_length,
			vocab_size=text_vocab_size,
			transformer_width=text_transformer_width,
			transformer_heads=text_transformer_heads,
			transformer_layers=text_transformer_layers,
		)

		if backbone_path:
			ckpt = torch.load(backbone_path)
			self.load_state_dict(ckpt, strict=False)


		with open(text_prompt_classes_path, 'r') as f:
			classes = f.read().strip().split('\n')

		self.prompt_learner = TextPromptLearner(
			classnames=classes,
			text_model=self.textual,
			num_prompts=text_num_prompts,
			prompts_init=text_prompt_init,
			CSC=text_prompt_CSC,
			ctx_pos=text_prompt_pos
		)
		self.tokenized_prompts = self.prompt_learner.tokenized_prompts

		with open(text_prompt_classes_path_atomic, 'r') as f:
			classes_atomic = f.read().strip().split('\n')

		self.prompt_learner_atomic = TextPromptLearner(
			classnames=classes_atomic,
			text_model=self.textual,
			num_prompts=text_num_prompts,
			prompts_init=text_prompt_init,
			CSC=text_prompt_CSC,
			ctx_pos=text_prompt_pos
		)
		self.tokenized_prompts_atomic = self.prompt_learner_atomic.tokenized_prompts

		self.face_backbone = resnet_scene.resnet18(pretrained=True)
		self.face_proj = nn.Linear(512, 768)
		self.head_proj = nn.Linear(784, 768)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.avgpool = nn.AvgPool2d(7)
		self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		self.heat_decoder = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 2),
		)
		self.emb_proj = nn.Linear(768 + 768, 512)
		self.ln_fff= LayerNorm(512)
		self.conv_inout = nn.Sequential(
			nn.Conv2d(768+768, 512, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(1),
			nn.ReLU(inplace=True)
		)
		self.fc_inout = nn.Linear(196, 1)

		# freeze encoders
		#self._freeze_visual_except_prompts_time_embed()
		self._freeze_textual()
		self.sigmoid = nn.Sigmoid()
		self.at_fc = nn.Linear(768,512)

		self.face_fc = nn.Linear(768, 768)
		self.feature_fc = nn.Linear(768*2, 768)

		self.atomic_trans = ITTrans(768,768,40)

		self.event_trans = ITTrans(768, 768, 20)
		self.interact_trans = ITTrans(768, 768, 20)
		self.log_vars = nn.Parameter(torch.zeros(4))


	def _freeze_visual_except_prompts_time_embed(self):
		for name, param in self.visual.named_parameters():
			if 'summary' in name or 'local' in name or 'global' in name or 'time_embed' in name:
				pass
			else:
				param.requires_grad = False

	def _freeze_textual(self):
		for name, param in self.textual.named_parameters():
			param.requires_grad = False

	def _decode_gaze(self,person,scene_feature):
		a1 = repeat(person.unsqueeze(1), 'b 1 d -> b n d', n=196)
		heat = torch.cat((scene_feature, a1), dim=-1)

		inout = rearrange(heat.float(), 'b (h w) d  -> b d h w', d=768 + 768, h=14, w=14)
		inout = self.conv_inout(inout)
		inout = inout.view(-1, 14 * 14)
		inout = self.fc_inout(inout)  # size: (B T),1

		heat = self.emb_proj(heat)
		heat = self.ln_fff(heat)
		heat = heat.mean(dim=1)

		heat = self.heat_decoder(heat)
		heat = self.sigmoid(heat)

		return heat,inout


	def forward(self, x,face,head):
		B, T, C, H, W = x.size()
		B, T, N, C, H, W = face.size()

		face = rearrange(face, 'b t n c h w -> (b t n) c h w', b=B, t=T, n=N, c=C, h=H, w=W)
		head = rearrange(head, 'b t n c h w -> (b t n) c h w', b=B, t=T, n=N, c=1, h=H, w=W)

		face_feature = self.face_backbone(face)
		head_emb = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
		head_emb = self.head_proj(head_emb)
		face_prompt = self.avgpool(face_feature).view(-1, 512)
		face_prompt = self.face_proj(face_prompt)
		face_token = face_prompt + head_emb
		face_token = rearrange(face_token, '(b t n) m -> (b t) n m', b=B, t=T, n=N)

		video_features,atomic_1,atomic_2,scene_feature, = self.visual(x,face_token)

		prompts_atomic = self.prompt_learner_atomic()
		tokenized_prompts_atomic = self.tokenized_prompts_atomic
		text_features_atomic = self.textual(prompts_atomic, tokenized_prompts_atomic)

		prompts = self.prompt_learner()
		tokenized_prompts = self.tokenized_prompts
		text_features = self.textual(prompts, tokenized_prompts)

		face_tokens = torch.cat((atomic_1.unsqueeze(1), atomic_2.unsqueeze(1)), dim=1)
		face_tokens = rearrange(face_tokens, '(b t) n e -> b (t n) e ', b=B, t=T, n=2)

		tem_tokens = self.atomic_trans(face_tokens,face_tokens,face_tokens)
		tem_face = tem_tokens  # [B:,:,:]
		tem_face = rearrange(tem_face, 'b (t n) e -> (b t) n e', b=B, t=T, n=2)

		atomic_1 = tem_face[:, 0, :]
		atomic_2 = tem_face[:, 1, :]
		f1 = atomic_1
		f2 = atomic_2

		int_p = torch.cat((self.face_fc(atomic_1), self.face_fc(atomic_2)), dim=1)
		int_p = self.feature_fc(int_p).unsqueeze(1)
		int_p = rearrange(int_p, '(b t) n e -> (b n) t e', b=B, t=T, n=1)

		video_features = video_features.view(B, T, 768)
		video_features = self.event_trans(video_features, video_features, video_features)
		video_features = self.interact_trans(video_features.view(B, T, 768), int_p, int_p)

		video_features = video_features.view(B * T, 1, 768).squeeze(1)

		video_features = video_features @ self.visual.proj
		video_features = rearrange(video_features, '(b t) e -> b t e', b=B, t=T).mean(dim=1)

		a1_heat,a1_inout = self._decode_gaze(f1,scene_feature)
		a2_heat, a2_inout = self._decode_gaze(f2, scene_feature)

		inout_pre = torch.cat((a1_inout,a2_inout),dim=-1).view(-1,1) #size: (B T 2),1

		heatmap_out = torch.cat((a1_heat.unsqueeze(1),a2_heat.unsqueeze(1)),dim=1)#size: (B T) 2 2
		heatmap_out = rearrange(heatmap_out.float(), 'b n l  -> (b n) l')#size: (B T 2) 2

		# normalized features
		video_features = video_features / video_features.norm(dim=-1, keepdim=True)
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)

		atomic_1 = self.at_fc(atomic_1)  # @self.visual.proj
		atomic_1 = rearrange(atomic_1, '(b t) e -> b t e', b=B, t=T)
		atomic_2 = self.at_fc(atomic_2)  # @ self.visual.proj
		atomic_2 = rearrange(atomic_2, '(b t) e -> b t e', b=B, t=T)
		atomic_1 = atomic_1 / atomic_1.norm(dim=-1, keepdim=True)
		atomic_2 = atomic_2 / atomic_2.norm(dim=-1, keepdim=True)
		text_features_atomic = text_features_atomic / text_features_atomic.norm(dim=-1, keepdim=True)

		# cosine similarity as logits
		logit_scale = self.logit_scale.exp()
		logits = logit_scale * video_features @ text_features.t()

		logit_scale_at = self.logit_scale_at.exp()
		logits_at1 = logit_scale_at * atomic_1 @ text_features_atomic.t()
		logits_at2 = logit_scale_at * atomic_2 @ text_features_atomic.t()

		logits_at1 = rearrange(logits_at1, 'b t c -> (b t) 1 c', b=B, t=T)
		logits_at2 = rearrange(logits_at2, 'b t c -> (b t) 1 c', b=B, t=T)

		logits_a = torch.cat((logits_at1,logits_at2),dim=1)
		logits_a = rearrange(logits_a, 'b n c -> (b n) c')

		return logits,logits_a,inout_pre,heatmap_out,self.log_vars


