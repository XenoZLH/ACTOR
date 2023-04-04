import mindspore
import mindspore.numpy as np
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from .Transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

# ACTOR Transformer Encoder
class Encoder_TRANSFORMER(nn.Cell):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation=nn.GELU, **kargs):
        super(Encoder_TRANSFORMER, self).__init__()
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
        self.num_heads = num_heads  # 注意力头数
        self.dropout = dropout  # dropout

        self.ablation = ablation  # 是否测试
        self.activation = activation  # 激活函数
        self.input_feats = self.njoints * self.nfeats
        
        self.muQuery = Parameter(ops.standard_normal((self.num_classes, self.latent_dim)), name='muQuery')
        self.sigmaQuery = Parameter(ops.standard_normal((self.num_classes, self.latent_dim)), name='sigmaQuery')
        
        self.skelEmbedding = nn.Dense(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        # 生成transformer Encoder
        TransEncoderLayer = TransformerEncoderLayer(dim=self.latent_dim,
                                                    num_heads=self.num_heads,
                                                    mlp_dim=self.ff_size,
                                                    keep_prob=1-self.dropout,
                                                    activation=self.activation)
        self.seqTransEncoder = TransformerEncoder(TransEncoderLayer, num_layers=self.num_layers)
    def construct(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = ops.permute(x, (3, 0, 1, 2))
        x = ops.reshape(x, (nframes, bs, njoints*nfeats))
        # embedding of the skeleton
        x = self.skelEmbedding(x)
        # adding the mu and sigma queries
        xseq = ops.concat((self.muQuery[y][None], self.sigmaQuery[y][None], x), dim=0)
        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)
        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = np.ones((bs, 2),  dtype=bool)
        maskseq = ops.concat((muandsigmaMask, mask), axis=1)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]
        return {"mu": mu, "logvar": logvar}
      
      
