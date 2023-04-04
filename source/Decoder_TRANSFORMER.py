import mindspore
import mindspore.numpy as np
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from .Transformer import TransformerDecoderLayer, TransformerDecoder

# ACTOR Transformer Decoder
class Decoder_TRANSFORMER(nn.Cell):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation=nn.GELU,**kargs):
        super(Decoder_TRANSFORMER, self).__init__()
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim  # 输入维数
        
        self.ff_size = ff_size  # 前馈网络隐含层数
        self.num_layers = num_layers  # 注意力层数
        self.num_heads = num_heads # 注意力头数
        self.dropout = dropout  # dropout

        self.ablation = ablation

        self.activation = activation  # 激活函数
        self.input_feats = self.njoints * self.nfeats
        
        self.actionBiases = Parameter(ops.standard_normal((self.num_classes, self.latent_dim)), name="actionBiases")
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        # 生成transformer Decoder
        seqTransDecoderLayer = nn.TransformerDecoderLayer(dim=self.latent_dim,
                                                          num_heads=self.num_heads,
                                                          mlp_dim=self.ff_size,
                                                          keep_prob=1-self.dropout,
                                                          activation=self.activation)
        elf.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        self.finallayer = nn.Dense(self.latent_dim, self.input_feats)
    def construct(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats
        z = z + self.actionBiases[y]
        z = z[None]
        timequeries = np.zeros((nframes, bs, latent_dim))
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z, tgt_key_padding_mask=~mask)
        output = self.finallayer(output)
        output = ops.reshape(output, (nframes, bs, njoints, nfeats))
        # zero for padded area
        output[~mask.T] = 0
        output = ops.permute(output, (1, 2, 3, 0))
        batch["output"] = output
        return batch
