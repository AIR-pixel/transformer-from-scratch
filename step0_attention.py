"""
阶段 0：自注意力 —— 先把形状搞清楚

本脚本不训练任何东西。它的唯一目的是把这一段可视化出来：
    论文里那个叫 "Scaled Dot-Product Attention" 的方框，
    落到代码里到底是什么。

读法建议：先不看代码，直接跑一遍，看输出。
看输出的时候盯住每一行的形状，尤其是 (B, T, T) 那一步。

维度都取很小的值（T=5, d_model=8），因为这样矩阵能完整打印出来，
可以用眼睛核对每一个数字，而不是只能相信它"应该是对的"。

运行：
    python step0_attention.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出保险

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)  # 固定随机种子，保证每次跑结果一样

# ---------------------------------------------------------------- 维度设定
B = 2   # batch size：一次喂几段文本
T = 5   # sequence length：每段文本几个 token
d_model = 8   # 每个 token 的向量维度（论文 base 版是 512）
n_head = 2    # 注意力头数（论文 base 版是 8）
d_head = d_model // n_head  # 每个头分到多少维（论文里是 64）

print("=" * 66)
print("维度设定")
print("=" * 66)
print(f"  B={B} (batch), T={T} (序列长度), d_model={d_model}, n_head={n_head}, d_head={d_head}")
print()
print("  可以把输入想成：2 句长度 5 的话，每个词用 8 个数表示。")
print("  注意 d_model 必须能被 n_head 整除，否则多头拆不匀。")


def show(name, tensor):
    """打印名字和形状。'1x5x8' 这种读法：batch 维 x 序列维 x 特征维。"""
    shape = " x ".join(str(s) for s in tensor.shape)
    print(f"  {name:<32}{shape}")


def head(title):
    print()
    print("-" * 66)
    print(title)
    print("-" * 66)


# ------------------------------------------------------------ 造一个假输入
head("第 1 步：输入")

x = torch.randn(B, T, d_model)
show("输入 x", x)
print()
print("  (B, T, 8) 的意思是：B 句话，每句 T 个 token，每个 token 是一串 8 维向量。")
print("  真实场景里，这 8 维来自 embedding 查表，这里是随机数，不影响形状演示。")


# ------------------------------------------------------- 第 2 步：Q K V
head("第 2 步：三个线性层，产生 Q、K、V")

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

q = W_q(x)   # Query：我在找什么
k = W_k(x)   # Key  ：我是什么
v = W_v(x)   # Value：我实际携带的信息

show("Q = x @ W_q", q)
show("K = x @ W_k", k)
show("V = x @ W_v", v)
print()
print("  关键：形状一个都没变，还是 (B, T, d_model)。")
print("  这三个线性层的作用只是把输入投影到三个不同的子空间，")
print("  让 '找什么' / '是什么' / '给什么' 可以各自学习。")
print("  论文里的 Q K V 就是这么来的，没有任何魔法。")


# ------------------------------------------- 第 3 步：算相似度（核心一步）
head("第 3 步：Q 和 K 做点积 —— 这一步产生 T x T 矩阵")

# 先做「单头」版本，也就是论文里最简单的那种注意力。
# 为什么是 Q @ K^T：
#   Q 是 (T, d)，K^T 是 (d, T)，乘完得到 (T, T)。
#   第 i 行第 j 列 = 第 i 个 token 的 query 和第 j 个 token 的 key 的内积，
#   也就是「第 i 个位置有多想关注第 j 个位置」——这就是 attention 的全部精髓。
scores = q @ k.transpose(-2, -1)  # transpose(-2,-1) 交换最后两维

show("q @ k^T (未缩放)", scores)
print()
print("  看最后两维：5 x 5。每一行是一个位置对全部 5 个位置的关注度打分。")

# 为什么要除以 sqrt(d_head)：
#   d 越大，点积的方差越大，softmax 会被推到极端（一个位置吃掉全部权重），
#   梯度会消失。除以 sqrt(d) 把方差拉回 1 左右。这是一行除以常数的操作，
#   论文里叫 scaled，之所以是 sqrt(d) 而不是 d，是因为内积的方差正比于 d。
scaled = scores / (d_model ** 0.5)
show("除以 sqrt(d_k) 后", scaled)
print(f"  这里 d_k = d_model = {d_model}，所以除以 sqrt({d_model}) = {d_model ** 0.5:.3f}")
print("  论文写的是 sqrt(d_k)：d_k 是 Q/K 的维度。单头时它就是整个 d_model；")
print("  多头时每个头的 d_k 变成 d_model/h，除的数也跟着变小——下面第 7 步会看到。")


# ------------------------------------------------- 第 4 步：softmax 成权重
head("第 4 步：softmax —— 把打分变成加起来等于 1 的权重")

attn = F.softmax(scaled, dim=-1)  # 沿最后一维归一化，也就是「对每个 query 归一化」

show("注意力权重", attn)
print()
print("  权重矩阵（每行和 = 1）：")
for i in range(T):
    row = "  ".join(f"{v:6.3f}" for v in attn[0, i].tolist())
    print(f"    位置 {i} 关注 -> {row}")
print()
print("  读法：第 i 行表示「位置 i 把注意力分给了哪些位置」。")
print("  现在每行都是均匀的一堆数字，因为 Q K 是随机初始化的，还没学。")

row_sums = attn.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "每行和应该是 1"
print(f"  [验证] 每行和 = {[round(v, 5) for v in row_sums[0].tolist()]} —— 都是 1，softmax 生效了。")


# ----------------------------------------------------- 第 5 步：乘上 V
head("第 5 步：用权重去加权求和 V")

out_single = attn @ v
show("attn @ V", out_single)
print()
print("  形状回到 (B, T, d_model)。")
print("  含义：每个位置输出的新向量 = 它关注的其它位置的信息的加权平均。")
print("  到这一步，'每个 token 都能看到其它所有 token' 这件事就完成了。")


# ------------------------------------------------------ 第 6 步：因果 mask
head("第 6 步：因果 mask —— 让位置 i 看不到它后面的位置")

print("  为什么需要它：GPT 是「预测下一个词」，如果位置 3 能看到位置 4，")
print("  那就等于考试时偷看了答案。所以要把右上角全部堵死。")
print()

mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # tril = 下三角，True 表示保留
print("  mask 长这样（True = 允许看，False = 禁止看）：")
for i in range(T):
    row = "  ".join("  T " if mask[i, j] else "  ." for j in range(T))
    print(f"    {row}")
print()

scores_masked = scaled.masked_fill(~mask, float("-inf"))  # 把禁止的位置设成负无穷
attn_masked = F.softmax(scores_masked, dim=-1)            # softmax 后负无穷自动变 0

print("  加上 mask 之后的注意力权重：")
for i in range(T):
    row = "  ".join(f"{v:6.3f}" for v in attn_masked[0, i].tolist())
    print(f"    位置 {i} 关注 -> {row}")
print()
print("  第 0 行：[1.000, 0, 0, 0, 0] —— 位置 0 是第一个词，它除了自己没别人可看。")
print("  第 4 行：全部有值 —— 最后一个位置可以看到前面所有词。")
print("  这就是 GPT 的工作方式：越靠后的位置，能看到的历史越长。")

future_leak = attn_masked.triu(diagonal=1).abs().max().item()
assert future_leak == 0.0, f"未来位置的权重必须是 0，实测 {future_leak}"
print(f"  [验证] 上三角最大权重 = {future_leak} —— 未来被彻底堵死了。")


# --------------------------------------------------- 第 7 步：多头是什么
head("第 7 步：多头注意力 —— 只是把上面这套并排跑 h 次")

print("  为什么多头：一次注意力只能学一种「关注方式」（比如只能学主谓关系）。")
print("  把头拆成 h 份，每个头独立学一种关系，最后拼起来。")
print("  注意：拆头不增加参数量，d_model 被 h 等分，总宽度不变。")
print()

q_multi = q.reshape(B, T, n_head, d_head).transpose(1, 2)
k_multi = k.reshape(B, T, n_head, d_head).transpose(1, 2)
v_multi = v.reshape(B, T, n_head, d_head).transpose(1, 2)

show("q 拆头后", q_multi)
print("  reshape 把最后一维 8 劈成 2 x 4（n_head x d_head），")
print("  transpose 把头维挪到前面，这样最后两维就永远是 (T, d_head)，")
print("  矩阵乘法就不用管头的事，一次算完所有头。")
print()

scores_m = q_multi @ k_multi.transpose(-2, -1)
show("多头分数", scores_m)
print("  最后两维还是 5 x 5，但前面多了个头维：2 个头各算各的 5x5。")
print()

scores_m = scores_m / (d_head ** 0.5)
scores_m = scores_m.masked_fill(~mask, float("-inf"))
attn_m = F.softmax(scores_m, dim=-1)
out_m = attn_m @ v_multi
show("多头输出", out_m)
print()

out_m = out_m.transpose(1, 2).reshape(B, T, d_model)  # 把头顶回原位，拼回去
show("拼回头之后", out_m)
print("  transpose 回去再 reshape，两个头拼成一个 8 维向量。")
print("  形状又回到 (B, T, d_model) —— 和输入完全一样。")
print("  这就是为什么 Transformer 能一层层往上堆：进出口形状不变。")


# --------------------------------------------- 第 8 步：装成一个正经模块
head("第 8 步：把它封成一个模块（阶段 1 直接复用）")


class MultiHeadAttention(nn.Module):
    """多头自注意力。构造函数占几行，forward 里的形状变换是全部难点。"""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = n_head
        self.d_head = d_model // n_head
        # 论文的做法：Q K V 各一个线性层。也可以合成一个大矩阵，效果一样。
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 拼回头之后的输出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        B, T, C = x.shape

        q = self.w_q(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v                                        # (B, h, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)           # 拼回头 -> (B, T, C)
        return self.w_o(out)                                 # 过输出投影


torch.manual_seed(0)  # 重新播种，保证模块里的权重和下面手动算的一致
mha = MultiHeadAttention(d_model, n_head)

# 用模块的权重，手动重算一遍上面第 7 步的流程
q2 = mha.w_q(x).reshape(B, T, n_head, d_head).transpose(1, 2)
k2 = mha.w_k(x).reshape(B, T, n_head, d_head).transpose(1, 2)
v2 = mha.w_v(x).reshape(B, T, n_head, d_head).transpose(1, 2)
a2 = (q2 @ k2.transpose(-2, -1)) / (d_head ** 0.5)
a2 = a2.masked_fill(~mask, float("-inf"))
a2 = F.softmax(a2, dim=-1)
o2 = a2 @ v2
o2 = o2.transpose(1, 2).reshape(B, T, d_model)
out_manual = mha.w_o(o2)

out_module = mha(x, causal=True)

show("手写流程的输出", out_manual)
show("模块 forward 的输出", out_module)
diff = (out_manual - out_module).abs().max().item()
assert diff < 1e-6, f"两者应该完全一致，实测最大差 {diff}"
print(f"  [验证] 两者最大差值 = {diff:.2e} —— 一致，模块写对了。")


# ---------------------------------------------------------- 收尾：结论
print()
print("=" * 66)
print("跑完了。到这里应该能确认这几件事：")
print("=" * 66)
print("  1. 注意力输入输出形状相同 -> 所以能一层层堆，堆 N 层不变形")
print("  2. 核心运算是 Q @ K^T -> 得到 T x T 矩阵，这是 O(T^2) 的唯一来源")
print("  3. mask 的作用是把右上角设成负无穷 -> GPT 靠它不偷看未来")
print("  4. 多头只是把同一套运算并排跑 h 次 -> 不增加参数量")
print("  5. 复杂度不在代码长度上：整个模块 30 行，难的是形状对齐")
print()
print("  阶段 1 会把这个模块乘以 N，加上 embedding 和输出头，变成能训练的 GPT。")
