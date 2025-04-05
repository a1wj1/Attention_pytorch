import torch
import torch.nn as nn
from attention import TransformerBlock

# 设置随机种子以便于结果复现
torch.manual_seed(0)

# 定义多头注意力层的参数
embed_dim = 64  # 嵌入的维度
num_heads = 4  # 注意力头的数量
batch_size = 5  # 批量大小
seq_length = 10  # 序列长度
latent_blocks = 4
dropout = 0.1
att_dropout = 0.2
att_heads = 4
cross_heads = 2
num_patches = 64
# 实例化多头注意力层
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

transformer = TransformerBlock(opt_dim=embed_dim,
                               latent_blocks=latent_blocks,
                               dropout=dropout,
                               att_dropout=att_dropout,
                               att_heads=att_heads,
                               cross_heads=cross_heads)

# 创建一些随机数据来模拟输入 (tokens, batch_size, emb_dim)
query = torch.rand((seq_length, batch_size, embed_dim))
key = torch.rand((seq_length, batch_size, embed_dim))
value = torch.rand((seq_length, batch_size, embed_dim))

# 应用transformer
transformer_attn = transformer(key, query)
print("transformer Output Shape: ", transformer_attn.shape)

# 应用多头注意力
attn_output, attn_output_weights = mha(query, key, value)

# 打印输出的形状
print("Attention Output Shape: ", attn_output.shape)
print("Attention Weights Shape: ", attn_output_weights.shape)