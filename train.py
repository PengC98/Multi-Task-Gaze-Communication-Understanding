from model import GazeCLIP

import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from dataset.dataset_utils import get_multi_hot_map, get_auc,get_heatmap_peak_coords,get_l2_dist,get_angular_error
from dataset.dataset import *
from dataset.vocation_dataset import *
from dataset.dy_gaze_dataset import *
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
from einops import rearrange, repeat
from sklearn.metrics import confusion_matrix, classification_report,average_precision_score
from torch.autograd import Variable
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
import random
random.seed(1)
def get_random_seed(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def uncertainty_weighted_loss(losses, log_vars):
	"""
	Computes the uncertainty-weighted loss.
	Args:
		losses (list of tensors): Losses from each task.
		log_vars (tensor): Learnable log variance parameters#.

	Returns:
		torch.Tensor: Total weighted loss.
	"""
	precisions = torch.exp(-log_vars)  # 
	weighted_losses = precisions * losses + log_vars  # Apply weighting
	return weighted_losses.sum()  # Sum over all tasks

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Train:
	def __init__(self, args):

		self.mode = args.mode
		self.train_continue = args.train_continue

		self.scope = args.scope
		self.dir_checkpoint = args.dir_checkpoint
		self.dir_log = args.dir_log
		self.accumulation_steps = args.accumulation_steps

		self.dir_data = args.dir_data
		#self.dir_result = args.dir_result

		self.num_epoch = args.num_epoch
		self.batch_size = args.batch_size

		self.lr_G = args.lr_G
		self.lr_D = args.lr_D

		self.wgt_c_a = args.wgt_c_a
		self.wgt_c_b = args.wgt_c_b

		self.wgt_pore = args.wgt_pore
		self.wgt_po = args.wgt_po

		self.beta1 = args.beta1

		self.gpu_ids = args.gpu_ids

		self.num_freq_disp = args.num_freq_disp
		self.num_freq_save = args.num_freq_save

		self.name_data = args.name_data
		get_random_seed(1024)
		self.train_dataset = self.select_dataset(self.name_data, 'train')
		#self.val_dataset = self.select_dataset(self.name_data, 'val')
		self.test_dataset = self.select_dataset(self.name_data, 'test')
		#train_size_l = int(0.25 * len(self.train_dataset))
		#train_size_n = len(self.train_dataset) - train_size_l
		#self.train_dataset, self.train_dataset_sp = torch.utils.data.random_split(self.train_dataset, [train_size_l, train_size_n])

		if self.gpu_ids and torch.cuda.is_available():
			self.device = torch.device("cuda:%d" % self.gpu_ids[0])
			torch.cuda.set_device(self.gpu_ids[0])
		else:
			self.device = torch.device("cpu")

		#self.pose_model = WHENet('WHENet\\WHENet.h5')

	def save(self, dir_chck, scene_net, optimG, epoch):
		if not os.path.exists(dir_chck):
			os.makedirs(dir_chck)

		torch.save({'scene_net': scene_net.state_dict(),
					'optimG': optimG.state_dict(),},
				   '%s/model_epoch%04d.pth' % (dir_chck, epoch))
	def select_dataset(self,name, mode):
		if name == 'GazeFollow':
			if mode=='train':
				labels = os.path.join(self.dir_data, "train_annotations_release.txt")
				dataset = GazeFollow(self.dir_data, labels, input_size=224, output_size=56,is_test_set=False)
			else:
				labels = os.path.join(self.dir_data, "test_annotations_release.txt")
				dataset = GazeFollow(self.dir_data, labels, input_size=224, output_size=56,is_test_set=True)
		if name == 'Videoattention':
			if mode=='train':
				labels = os.path.join(self.dir_data, "annotations/train")
				dataset = VideoAttentionTargetImages(self.dir_data, labels, input_size=224, output_size=56,is_test_set=False)
			else:
				labels = os.path.join(self.dir_data, "annotations/test")
				dataset = VideoAttentionTargetImages(self.dir_data, labels, input_size=224, output_size=56,is_test_set=True)

		if name == 'Vocation':
			labels = os.path.join(self.dir_data, "event",mode)
			dataset = Vocation(self.dir_data, labels, input_size=224, output_size=56,sampling_rate = -1,num_frames = 20, train_mode=mode)

		if name == 'DyGaze':
			labels = os.path.join(self.dir_data, "event", mode)
			dataset = DYgaze(self.dir_data, labels, input_size=224, output_size=56, sampling_rate=-1, num_frames=20,
							   train_mode=mode)
		return dataset

	def load(self, dir_chck,scene_net, optimG=[], epoch=[], mode='train'):
		if not epoch:
			ckpt = os.listdir(dir_chck)
			ckpt.sort()
			epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])
			print(epoch)

		dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

		print('Loaded %dth network' % epoch)

		if mode == 'train':
			scene_net.load_state_dict(dict_net['scene_net'])
			optimG.load_state_dict(dict_net['optimG'])

			return scene_net, optimG, epoch

		elif mode == 'test':
			#plg.load_state_dict(dict_net['plg'])
			scene_net.load_state_dict(dict_net['scene_net'])

			return scene_net, epoch


	def train(self):
		mode = self.mode

		train_continue = self.train_continue
		num_epoch = self.num_epoch
		accumulation_steps = self.accumulation_steps

		lr_G = self.lr_G


		batch_size = self.batch_size
		device = self.device

		gpu_ids = self.gpu_ids

		name_data = self.name_data

		num_freq_disp = self.num_freq_disp
		num_freq_save = self.num_freq_save

		idx_tensor = [idx for idx in range(66)]
		idx_tensor = torch.FloatTensor(idx_tensor).to(device)

		## setup dataset
		dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

		dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')

		loader_train = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
		num_train = len(self.train_dataset)

		num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

		scene_net = GazeCLIP.GazeCLIP(device, backbone_path='clip_pretrained.pth', input_size=(224, 224), num_frames=20,
									  feature_dim=768, patch_size=(16, 16), num_heads=12, num_layers=12, mlp_factor=4.0,
									  embed_dim=512, text_context_length=77, text_vocab_size=49408,
									  text_transformer_width=512, text_transformer_heads=8, text_transformer_layers=12,
									  text_num_prompts=15, text_prompt_pos='end', text_prompt_init='',
									  text_prompt_CSC=True, text_prompt_classes_path='dataset/event_name.txt',
									  text_prompt_classes_path_atomic='dataset/atomic_name.txt',
									  zeroshot_evaluation=False, zeroshot_text_features_path='').to(device)
		#init_net(scene_net, init_type='normal', init_gain=0.05, gpu_ids=gpu_ids)
	   # init_net(plg, init_type='normal', init_gain=0.05, gpu_ids=gpu_ids)

		mse_loss=nn.MSELoss(reduction='none').to(device)
		event_loss = FocalLoss(class_num=5)
		atomic_loss = torch.nn.CrossEntropyLoss()
		inout_loss = nn.BCEWithLogitsLoss()
		#atomic_loss.to(device)


		paramsG_scene = filter(lambda p: p.requires_grad, scene_net.parameters())
		optimG = torch.optim.AdamW(paramsG_scene, lr=lr_G, weight_decay=0.001)
		schedG = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, T_max=40)
		#optimG = torch.optim.Adam(paramsG_scene, lr=lr_G)#,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
		#schedG = torch.optim.lr_scheduler.StepLR(optimG,step_size=10, gamma=0.1)


		## load from checkpoints
		st_epoch = 0

		if train_continue == 'on':
			scene_net,optimG, st_epoch = \
				self.load(dir_chck, scene_net, optimG, mode=mode)

		## setup tensorboard
		writer_train = SummaryWriter(log_dir=dir_log_train)


		for epoch in range(st_epoch + 1, num_epoch + 1):
			#schedG.step()
			#atomic_loss._hook_before_epoch(epoch)


			scene_net.train()

			loss_event_train = []
			loss_atomic_train = []

			loss_heat_train = []
			loss_inout_train = []


			for i, data in enumerate(loader_train, 1):

				def should(freq):
					return freq > 0 and (i % freq == 0 or i == num_batch_train)

				head = data['head'].to(device)
				rgb_img = data['img'].to(device)
				event_label = data['event_label'].to(device)
				atomic_label = data['atomic_label'].to(device)
				gaze_inside = data['gaze_inside'].to(device)
				gaze_point = data['gaze_point'].to(device)
				face = data['face'].to(device)
				head = head.float()


				#t_heatmap = rearrange(t_heatmap, 'b t n h w  -> (b t n) h w')
				gaze_point = rearrange(gaze_point, 'b t n l  -> (b t n) l')
				atomic_label = rearrange(atomic_label, 'b t n  -> (b t n) 1').squeeze(1).long()
				gaze_inside = rearrange(gaze_inside, 'b t n  -> (b t n) 1')
				event_label = event_label.squeeze(1).long()


				event_logits,atomic_logits,inout_pre,heat_pre,log_var = scene_net(rgb_img,face,head)



				loss_event = event_loss(event_logits,event_label)
				loss_atomic = atomic_loss(atomic_logits,atomic_label)
				loss_heat = mse_loss(heat_pre, gaze_point)*100
				loss_heat = torch.mean(loss_heat, dim=1)

				loss_heat = torch.mul(loss_heat, gaze_inside.squeeze())

				loss_heat = torch.sum(loss_heat) / (torch.sum(gaze_inside)+1e-9)

				loss_inout = inout_loss(inout_pre.squeeze(), gaze_inside.squeeze())*10

				optimG.zero_grad()

				losses = torch.stack([loss_event,loss_atomic,loss_heat,loss_inout])
				total_loss = uncertainty_weighted_loss(losses, log_var)


				# Compute final loss
				loss_G = total_loss


				loss_G.backward()
				optimG.step()

				loss_event_train += [loss_event.item()]
				loss_atomic_train+=[loss_atomic.item()]
				loss_heat_train += [loss_heat.item()]
				loss_inout_train += [loss_inout.item()]


				print('TRAIN:EPOCH %d: BATCH %04d/%04d: '
						'lossE: %.4f lossA: %.4f lossH: %.4f lossI: %.4f'
						% (epoch, i, num_batch_train, mean(loss_event_train),mean(loss_atomic_train),  mean(loss_heat_train), mean(loss_inout_train)))

				if should(num_freq_disp):

					del rgb_img
					del head
					del face
			schedG.step()


			writer_train.add_scalar('loss_event', mean(loss_event_train), epoch)
			writer_train.add_scalar('loss_atomic', mean(loss_atomic_train), epoch)
			writer_train.add_scalar('loss_heat', mean(loss_heat_train), epoch)
			writer_train.add_scalar('loss_inout', mean(loss_inout_train), epoch)

			if (epoch % num_freq_save) == 0:
				self.save(dir_chck, scene_net, optimG, epoch)


		writer_train.close()

	def evaluate(self,model,dataloader,device):
		tot, e_hit1, e_hit2, a_hit1, a_hit2,tota = 0, 0, 0, 0, 0, 0
		event_loss = FocalLoss(class_num=5)#torch.nn.CrossEntropyLoss()
		atomic_loss = FocalLoss(class_num=6)
		loss_event_train = []
		loss_atomic_train = []
		gaze_inside_all = []
		gaze_inside_pred_all = []
		event_all = []
		event_pred_all = []
		atomic_all = []
		atomic_pred_all = []
		avg_Dis = []

		for i, data in enumerate(dataloader, 1):
			head = data['head'].to(device)
			rgb_img = data['img'].to(device)
			event_label = data['event_label']
			atomic_label = data['atomic_label']
			gaze_inside = data['gaze_inside']
			t_heatmap = data['true_label_heatmap']
			gaze_coords = data['gaze_coord']
			face = data['face'].to(device)
			head = head.float()
			B, T, N, C, H, W = face.size()

			t_heatmap = rearrange(t_heatmap, 'b t n h w  -> (b t n) h w')
			atomic_label = rearrange(atomic_label, 'b t n  -> (b t n) 1')
			gaze_coords = rearrange(gaze_coords, 'b t n e  -> (b t n) e')
			gaze_inside = rearrange(gaze_inside, 'b t n  -> (b t n) 1')
			# atomic_score = torch.zeros((B * T * N, 6))
			with torch.no_grad():
				event_logits, atomic_logits, inout_pre, heat_pre, _ = model(rgb_img, face, head)
				event_score = event_logits.softmax(dim=-1).cpu()
				event_pre = torch.argmax(event_score, 1)
				atomic_score = atomic_logits.softmax(dim=-1).cpu()
				#a = torch.sum(atomic_score_pre[:,:4],dim=-1)
				#atomic_score[:,0] = a
				#atomic_score[:,1:] = atomic_score_pre[:,4:]
			atomic_pre = torch.argmax(atomic_score, 1)

			heat_pre = heat_pre.cpu()
			tota += atomic_label.shape[0]
			tot += event_label.shape[0]
			e_hit1 += (event_score.topk(1)[1] == event_label).sum().item()
			e_hit2 += (event_score.topk(2)[1] == event_label).sum().item()
			a_hit1 += (atomic_score.topk(1)[1] == atomic_label).sum().item()
			a_hit2 += (atomic_score.topk(2)[1] == atomic_label).sum().item()
			gaze_inside_all.extend(gaze_inside.squeeze().cpu().tolist())
			gaze_inside_pred_all.extend(inout_pre.squeeze().cpu().tolist())
			event_all.extend(event_label.cpu().tolist())
			event_pred_all.extend(event_pre.cpu().tolist())
			atomic_all.extend(atomic_label.cpu().tolist())
			atomic_pred_all.extend(atomic_pre.cpu().tolist())


			for gt_point, pred in zip(gaze_coords, heat_pre):
				valid_gaze = gt_point[gt_point != -1].view(-1, 2)
				if len(valid_gaze) == 0:
					continue

				norm_p = torch.tensor([pred[0], pred[1]])

				avg_distance = get_l2_dist(valid_gaze.squeeze(0), norm_p)
				# AUC.append(auc_score)
				avg_Dis.append(avg_distance)

		ap = average_precision_score(gaze_inside_all, gaze_inside_pred_all)
		print('gaze inside ap: %s' % ap)

		print(min(avg_Dis))
		print(np.mean(avg_Dis))
		print(
			f'Accuracy on val set: event top1={e_hit1 / tot * 100:.2f}%, top2={e_hit2 / tot * 100:.2f}% event top1={a_hit1 / tota * 100:.2f}%, top2={a_hit2 / tota * 100:.2f}% ')

		return e_hit1 / tot, a_hit1 / tota




	def test(self):
		mode = self.mode

		batch_size = self.batch_size
		device = self.device
		gpu_ids = self.gpu_ids
		name_data = self.name_data

		## setup dataset
		dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)


		transform_inv = transforms.Compose(
			[
				transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
				transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
				ToNumpy()
			]
		)

		loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

		num_test = len(self.test_dataset)

		num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

		scene_net = GazeCLIP.GazeCLIP(device, backbone_path='clip_pretrained.pth', input_size=(224, 224), num_frames=20,
									  feature_dim=768, patch_size=(16, 16), num_heads=12, num_layers=12, mlp_factor=4.0,
									  embed_dim=512, text_context_length=77, text_vocab_size=49408,
									  text_transformer_width=512, text_transformer_heads=8, text_transformer_layers=12,
									  text_num_prompts=15, text_prompt_pos='end', text_prompt_init='',
									  text_prompt_CSC=True, text_prompt_classes_path='dataset/event_name.txt',
									  text_prompt_classes_path_atomic='dataset/atomic_name.txt',
									  zeroshot_evaluation=False, zeroshot_text_features_path='').to(device)


		## load from checkpoints
		st_epoch = 0
		event_names = ['SingleGaze', 'GazeFollow', 'AvertGaze', 'MutualGaze', 'JointAtt']
		atomic_names = ['single', 'miss', 'void', 'mutual','share']
		NUM_e = len(event_names)
		NUM_a = len(atomic_names)
		scene_net, st_epoch = self.load(dir_chck, scene_net, mode=mode)
		AUC = []
		gaze_inside_all = []
		gaze_inside_pred_all = []
		event_all = []
		event_pred_all = []
		atomic_all = []
		atomic_pred_all = []
		avg_Dis = []
		min_Ang = []
		avg_Ang = []
		## test phase
		tot,tota, e_hit1, e_hit2, a_hit1, a_hit2 = 0, 0, 0, 0, 0,0
		with torch.no_grad():
			scene_net.eval()
			conf_matrix_event = torch.zeros(5, 5)
			atomic_matrix_event = torch.zeros(5, 5)
			for i, data in enumerate(loader_test, 1):
				head = data['head'].to(device)
				rgb_img = data['img'].to(device)
				event_label = data['event_label']
				atomic_label = data['atomic_label']
				gaze_inside = data['gaze_inside']
				t_heatmap = data['true_label_heatmap']
				gaze_coords = data['gaze_coord']
				face = data['face'].to(device)
				head = head.float()
				B, T, N, C, H, W = face.size()


				t_heatmap = rearrange(t_heatmap, 'b t n h w  -> (b t n) h w')
				atomic_label = rearrange(atomic_label, 'b t n  -> (b t n) 1')
				gaze_coords = rearrange(gaze_coords, 'b t n e  -> (b t n) e')
				gaze_inside = rearrange(gaze_inside, 'b t n  -> (b t n) 1')
				#atomic_score = torch.zeros((B * T * N, 6))
				with torch.no_grad():
					event_logits,atomic_logits, inout_pre, heat_pre,_ = scene_net(rgb_img, face, head)
					event_score = event_logits.softmax(dim=-1).cpu()
					event_pre = torch.argmax(event_score, 1)
					atomic_score = atomic_logits.softmax(dim=-1).cpu()
					#a = torch.sum(atomic_score_pre[:, :4], dim=-1)
					#atomic_score[:, 0] = a
					#atomic_score[:, 1:] = atomic_score_pre[:, 4:]

					atomic_pre = torch.argmax(atomic_score, 1)

					heat_pre = heat_pre.cpu()
				tota += atomic_label.shape[0]
				tot += event_label.shape[0]
				e_hit1 += (event_score.topk(1)[1] == event_label).sum().item()
				e_hit2 += (event_score.topk(2)[1] == event_label).sum().item()
				a_hit1 += (atomic_score.topk(1)[1] == atomic_label).sum().item()
				a_hit2 += (atomic_score.topk(2)[1] == atomic_label).sum().item()
				gaze_inside_all.extend(gaze_inside.squeeze().cpu().tolist())
				gaze_inside_pred_all.extend(inout_pre.squeeze().cpu().tolist())
				event_all.extend(event_label.cpu().tolist())
				event_pred_all.extend(event_pre.cpu().tolist())
				atomic_all.extend(atomic_label.cpu().tolist())
				atomic_pred_all.extend(atomic_pre.cpu().tolist())
				conf_matrix_event = confusion_matrix_t(event_logits,event_label.cpu().long(),conf_matrix_event)
				atomic_matrix_event = confusion_matrix_t(atomic_logits, atomic_label.cpu().long(), atomic_matrix_event)


				for gt_point, pred in zip(gaze_coords,  heat_pre):
					valid_gaze = gt_point[gt_point != -1].view(-1, 2)
					if len(valid_gaze) == 0:
						continue

					norm_p = torch.tensor([pred[0], pred[1]])


					avg_distance = get_l2_dist(valid_gaze.squeeze(0), norm_p)
					#AUC.append(auc_score)
					avg_Dis.append(avg_distance)
					

				#print('test_AUC_e: %s  , test_avg_Dis: %s' % (np.mean(AUC), np.mean(avg_Dis)))
			print(classification_report(np.array(event_all), np.array(event_pred_all), target_names= event_names, digits=4,labels=list(range(NUM_e))))
			print(classification_report(np.array(atomic_all), np.array(atomic_pred_all), target_names=atomic_names, digits=4,
											labels=list(range(NUM_a))))
			ap = average_precision_score(gaze_inside_all, gaze_inside_pred_all)
			print('gaze inside ap: %s' %ap)
			print(conf_matrix_event)
			print(atomic_matrix_event)
			print(min(avg_Dis))
			print(np.mean(avg_Dis))
			print(f'Accuracy on test set: event top1={e_hit1 / tot * 100:.2f}%, top2={e_hit2 / tot * 100:.2f}% event top1={a_hit1 / tota * 100:.2f}%, top2={a_hit2 / tota * 100:.2f}% ')



def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def confusion_matrix_t(preds, labels, conf_matrix):
	preds = torch.argmax(preds, 1)
	for p, t in zip(preds, labels):
		conf_matrix[p, t] += 1

	#true_label = labels.cpu().numpy()
	#pred_label = preds.cpu().numpy()
	#print(classification_report(true_label, pred_label, digits=4))
	return conf_matrix

def append_index(dir_result, fileset, step=False):
	index_path = os.path.join(dir_result, "index.html")
	if os.path.exists(index_path):
		index = open(index_path, "a")
	else:
		index = open(index_path, "w")
		index.write("<html><body><table><tr>")
		if step:
			index.write("<th>step</th>")
		for key, value in fileset.items():
			index.write("<th>%s</th>" % key)
		index.write('</tr>')

	# for fileset in filesets:
	index.write("<tr>")

	if step:
		index.write("<td>%d</td>" % fileset["step"])
	index.write("<td>%s</td>" % fileset["name"])

	del fileset['name']

	for key, value in fileset.items():
		index.write("<td><img src='images/%s'></td>" % value)

	index.write("</tr>")
	return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
	fig, ax = plt.subplots()

	ax.plot(output.transpose(1, 0).detach().numpy(), '-')
	ax.plot(label.transpose(1, 0).detach().numpy(), '--')

	ax.set_xlim(0, 400)

	ax.grid(True)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)

	writer.add_figure(namescope, fig, epoch)

class AG_loss(torch.nn.Module):
	def __init__(self):
		super(AG_loss, self).__init__()
		self.cosine_similarity = nn.CosineSimilarity()
	def forward(self,d1, d2):
		cos = self.cosine_similarity(d1, d2)

		loss = 1-cos
		return torch.mean(loss)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
	return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
	caption_loss = contrastive_loss(similarity)
	image_loss = contrastive_loss(similarity.t())
	return (caption_loss + image_loss) / 2.0


class HLoss(torch.nn.Module):
	def __init__(self):
		super(HLoss, self).__init__()

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, -1)
		x = torch.softmax(x, dim=-1)
		ind = torch.ones_like(x, requires_grad=True)
		ind = torch.div(ind , ind.sum(-1, keepdim=True))

		pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))
		pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

		# saliency value of region
		pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
		neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
		p = torch.cat([pos_x, neg_x], dim=-1)

		# pixel percentage of region
		pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
		neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
		ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

		b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1)
		b = -1.0 * b.sum(dim=-1)
		return b.mean()

class FocalLoss(nn.Module):
	r"""
		This criterion is a implemenation of Focal Loss, which is proposed in
		Focal Loss for Dense Object Detection.

			Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

		The losses are averaged across observations for each minibatch.

		Args:
			alpha(1D Tensor, Variable) : the scalar factor for this criterion
			gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
								   putting more focus on hard, misclassiﬁed examples
			size_average(bool): By default, the losses are averaged over observations for each minibatch.
								However, if the field size_average is set to False, the losses are
								instead summed for each minibatch.


	"""
	def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
		super(FocalLoss, self).__init__()
		if alpha is None:
			self.alpha = Variable(torch.ones(class_num, 1))
		else:
			if isinstance(alpha, Variable):
				self.alpha = alpha
			else:
				self.alpha = Variable(alpha)
		self.gamma = gamma
		self.class_num = class_num
		self.size_average = size_average

	def forward(self, inputs, targets):
		N = inputs.size(0)
		C = inputs.size(1)
		P = F.softmax(inputs)

		class_mask = inputs.data.new(N, C).fill_(0)
		class_mask = Variable(class_mask)
		ids = targets.view(-1, 1)
		class_mask.scatter_(1, ids.data, 1.)
		#print(class_mask)


		if inputs.is_cuda and not self.alpha.is_cuda:
			self.alpha = self.alpha.cuda()
		alpha = self.alpha[ids.data.view(-1)]

		probs = (P*class_mask).sum(1).view(-1,1)

		log_p = probs.log()
		#print('probs size= {}'.format(probs.size()))
		#print(probs)

		batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
		#print('-----bacth_loss------')
		#print(batch_loss)


		if self.size_average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()
		return loss



class LDAMLoss(nn.Module):
	def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=3):
		super().__init__()
		if cls_num_list is None:
			# No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
			self.m_list = None
		else:
			self.reweight_epoch = reweight_epoch
			m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
			m_list = m_list * (max_m / np.max(m_list))
			m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
			self.m_list = m_list
			assert s > 0
			self.s = s
			if reweight_epoch != -1:
				idx = 1  # condition could be put in order to set idx
				betas = [0, 0.9999]
				effective_num = 1.0 - np.power(betas[idx], cls_num_list)
				per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
				per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
				self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
			else:
				self.per_cls_weights_enabled = None
				self.per_cls_weights = None

	def to(self, device):
		super().to(device)
		if self.m_list is not None:
			self.m_list = self.m_list.to(device)

		if self.per_cls_weights_enabled is not None:
			self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

		return self

	def _hook_before_epoch(self, epoch):
		if self.reweight_epoch != -1:
			self.epoch = epoch

			if epoch > self.reweight_epoch:
				self.per_cls_weights = self.per_cls_weights_enabled
			else:
				self.per_cls_weights = None

	def get_final_output(self, output_logits, target):
		x = output_logits

		index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
		index.scatter_(1, target.data.view(-1, 1), 1)

		index_float = index.float()
		batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

		batch_m = batch_m.view((-1, 1))
		x_m = x - batch_m * self.s

		final_output = torch.where(index, x_m, x)
		return final_output

	def forward(self, output_logits, target):
		if self.m_list is None:
			return F.cross_entropy(output_logits, target)

		final_output = self.get_final_output(output_logits, target)
		return F.cross_entropy(final_output, target, weight=self.per_cls_weights)