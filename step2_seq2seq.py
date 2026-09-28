"""
阶段 2：序列反转 —— 补上论文的左半边

三个阶段的分工：
    阶段 0：注意力内部在算什么（一个模块）
    阶段 1：decoder-only 的 GPT（只用了论文 Figure 1 的右半边）
    阶段 2：完整的 encoder-decoder（论文 Figure 1 的完整形态）

这里新增三样阶段 1 没有的东西：
    1. Encoder —— 双向注意力，没有 mask，每个位置能看全句
    2. Cross-Attention —— Q 来自 decoder，K/V 来自 encoder 的输出
    3. 正弦位置编码 —— 论文原文的设计，顺便验证它到底有什么用

任务：把一串数字倒过来。
    "1 3 5 7" -> "7 5 3 1"

为什么用这个任务而不用语言：
    它有唯一正确答案，所以能用「准确率」而不是「loss 好不好看」来判断。
    语言任务的 loss 从 2.5 降到 2.3 你可能看不出来发生了什么，
    但准确率从 3% 涨到 98%，没有任何歧义。
    这就是「真懂没真懂」的检验方式。

运行：
    "C:/Program Files/ComfyUI-aki-v3/python/python.exe" step2_seq2seq.py
对比正弦编码与可学习编码：
    "C:/Program Files/ComfyUI-aki-v3/python/python.exe" step2_seq2seq.py --pos sinusoidal
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 词表
N_DIGITS = 10
PAD = 10          # 填充符：让同一批里的短序列补齐到一样长
BOS = 11          # 序列开始
EOS = 12          # 序列结束
VOCAB = 13

# 【踩坑记录 —— 这是这个脚本最值钱的一条注释，比模型本身值钱】
#
# 第一版我让张量长度 = 当前任务的 max_len，埋了一个非常隐蔽的雷：
#     训练时序列上限 8  -> 张量长度 9
#     评估「长度 4」时   -> 张量长度 5
# 于是同一个「长度 4 的句子」：训练时后面跟着 4 个 PAD，评估时一个 PAD 都没有。
# encoder 是双向注意力，它会真的「看到」这些 PAD 位置，
# attention 的 softmax 要在更多/更少的元素上归一化，输出的表示就飘了。
# 表现就是：训练时长度 4 学得好好的，换成短张量评估时准确率从 100% 掉到 49%。
# 而长度 8 因为两次都是「无 PAD」，侥幸 100% —— 这种「部分正确」最迷惑人。
#
# 教训：padding 结构是输入分布的一部分。训练/评估/生成必须用同一个缓冲区长度。
# 这就是论文里一句 "we pad sequences" 背后真正要处理的东西，论文一个字没写。
BUF = 12      # 缓冲区单侧容量，张量实际长度 BUF + 1
POS_LEN = 20  # 位置编码表大小，留足余量给生成时的自回归增长


# ============================================================== 数据
def gen_batch(batch_size, min_len, max_len, buf_len=None):
    """
    现场生成数据。这个任务不需要数据集——规则就是数据本身。

    返回三个张量：
        src   : (B, buf+1)  encoder 输入 = 数字序列 + EOS
        tgt_in: (B, buf+1)  decoder 输入 = BOS + 反转后的序列
        tgt_out(B, buf+1)   decoder 目标 = 反转后的序列 + EOS

    buf_len 是缓冲区长度，必须固定（见文件开头的踩坑记录）。
    实际序列长度在 min_len ~ max_len 之间随机，剩下的位置填 PAD。

    tgt_in 和 tgt_out 依然是错开一格的关系，和阶段 1 的自监督完全一样。
    区别在于：这里的目标不是"下一个字符"，而是"输入的倒序"——
    模型必须真的看懂整个输入句子，光靠语言模型的语感是猜不出来的。
    """
    if buf_len is None:
        buf_len = max_len

    L = torch.randint(min_len, max_len + 1, (batch_size,))

    src = torch.full((batch_size, buf_len + 1), PAD, dtype=torch.long)
    tgt_in = torch.full((batch_size, buf_len + 1), PAD, dtype=torch.long)
    tgt_out = torch.full((batch_size, buf_len + 1), PAD, dtype=torch.long)

    for i in range(batch_size):
        n = int(L[i])
        seq = torch.randint(0, N_DIGITS, (n,))
        rev = torch.flip(seq, dims=[0])

        src[i, :n] = seq
        src[i, n] = EOS

        tgt_in[i, 0] = BOS
        tgt_in[i, 1: n + 1] = rev

        tgt_out[i, :n] = rev
        tgt_out[i, n] = EOS

    return src, tgt_in, tgt_out


# ============================================================== 位置编码
def sinusoidal_encoding(T, d):
    """
    论文 3.5 节那个公式，原样翻译成代码：

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    为什么是正弦而不是学一个 embedding：
    论文的猜测是——不同频率的波形让模型能通过线性组合表达相对位置，
    而且能外推到训练时没见过的长度。后面这个说法一直被怀疑，
    但直到今天很多模型还在用它（或它的变体 RoPE）。
    在这个脚本里你可以亲手验证一下：训练时最长 8，测试时给 12，看看会不会崩。

    i 从 0 到 d/2，所以偶数维填 sin、奇数维填 cos，两两一组。
    """
    pe = torch.zeros(T, d)
    pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)        # (T, 1)
    # 10000^(-2i/d)，写成 exp/log 形式数值更稳
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)   # 偶数列
    pe[:, 1::2] = torch.cos(pos * div)   # 奇数列
    return pe


class PositionalEncoding(nn.Module):
    """可切换两种实现，好让你对比。"""

    def __init__(self, d_model, max_len, mode="learned"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            # 每个位置学一个向量。训练没覆盖到的位置就是随机的，外推必然失败。
            self.emb = nn.Embedding(max_len, d_model)
        else:
            # 正弦编码不参与训练（buffer 不会被优化器更新），是写死的常量。
            self.register_buffer("pe", sinusoidal_encoding(max_len, d_model), persistent=False)

    def forward(self, x):
        T = x.shape[1]
        if self.mode == "learned":
            pos = torch.arange(T, device=x.device)
            return x + self.emb(pos)
        return x + self.pe[:T]


# ============================================================== 注意力
class MHA(nn.Module):
    """
    一个模块搞定三种注意力。区别只在两个开关：

                                   memory    causal
        encoder 自注意力            None      False    <- 双向，能看全句
        decoder 自注意力            None      True     <- 不能看未来
        cross-attention             encoder输出 False    <- Q 来自 decoder，K/V 来自 encoder

    这就是论文 Figure 1 里那三种画法不同的注意力，代码上是同一个类。
    """

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, causal=False):
        B, T, C = x.shape
        kv = x if memory is None else memory       # 只有这一行区分 self / cross
        S = kv.shape[1]

        q = self.w_q(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(kv).reshape(B, S, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(kv).reshape(B, S, self.n_head, self.d_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        if causal:
            # 注意这里是 (T, S) 而不是方阵：cross-attention 时 T 和 S 可以不等长
            m = torch.tril(torch.ones(T, S, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~m, float("-inf"))

        att = self.dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.w_o(out)


class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================== 层
class EncoderLayer(nn.Module):
    """注意：这里的自注意力没有因果 mask，所以是双向的。

    双向意味着每个位置都能看到整句话——这是 encoder 的能力来源，
    也是为什么 encoder 不能做生成（生成时未来还不存在）。
    """

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), causal=False)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    """和阶段 1 的 Block 相比，多夹了一层 cross-attention。

    顺序是：自己看自己（带 mask）-> 看 encoder（不带 mask）-> 前馈。
    这个顺序不能乱：先把当前已有的上下文理清，再去查资料。
    """

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MHA(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MHA(d_model, n_head, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout)

    def forward(self, x, memory):
        x = x + self.self_attn(self.ln1(x), causal=True)     # 看自己（不许看未来）
        x = x + self.cross_attn(self.ln2(x), memory=memory)  # 看 encoder
        x = x + self.ffn(self.ln3(x))
        return x


# ============================================================== 模型
class Seq2Seq(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layer, max_len, dropout, pos_mode):
        super().__init__()
        self.max_len = max_len
        self.enc_emb = nn.Embedding(vocab, d_model)
        self.dec_emb = nn.Embedding(vocab, d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len, pos_mode)
        self.dec_pos = PositionalEncoding(d_model, max_len, pos_mode)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.dec_emb.weight      # 权重共享，同阶段 1

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, src):
        """
        encoder 只跑一次就够了——不管 decoder 生成到第几个字，
        它看的那句输入都是同一句。这也是为什么它只算一次。
        """
        x = self.drop(self.enc_pos(self.enc_emb(src)))
        for layer in self.encoder:
            x = layer(x)
        return x                                  # (B, S, d) 这就是 memory

    def decode(self, tgt_in, memory):
        x = self.drop(self.dec_pos(self.dec_emb(tgt_in)))
        for layer in self.decoder:
            x = layer(x, memory)
        return self.head(self.ln_f(x))            # (B, T, vocab)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src))

    @torch.no_grad()
    def greedy_decode(self, src, max_new_tokens=14):
        """
        真正的推理方式：一个一个往外吐，每吐一个就接回去当下一步的输入。

        注意这里每步都把整条前缀重新喂一遍 decoder。工程上会缓存 K/V 避免重算
        （就是 KV cache），但那样代码会长一倍。学习阶段先用最朴素的方式，
        看清"自回归"到底在干什么：生成 L 个字符 = 跑 L 次前向。
        """
        self.eval()
        B = src.shape[0]
        memory = self.encode(src)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)

        for _ in range(max_new_tokens):
            logits = self.decode(ys, memory)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt == EOS).all():
                break
        return ys


# ============================================================== 评估
@torch.no_grad()
def evaluate(model, max_len, batch_size=256, n_batches=8, seed=0):
    """
    用准确率而不是 loss 来判定。

    token 准确率：每个位置的字符猜对的比例
    整句准确率  ：整条序列完全正确（含 EOS）的比例 —— 这个才是真正的指标

    为什么两个都要看：token 准确率 90% 听起来不错，但如果每条序列有 8 个位置，
    0.9^8 ≈ 43%，整句只有四成对。生成任务里任何一处错都会污染后续，所以整句准确率才是真相。
    """
    torch.manual_seed(seed)
    # 评估必须关掉 dropout，否则准确率会随机抖动，看不出真实趋势
    was_training = model.training
    model.eval()
    tok_ok = tot_tok = 0
    seq_ok = tot_seq = 0

    for _ in range(n_batches):
        src, tgt_in, tgt_out = gen_batch(batch_size, 1, max_len, BUF)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        pred = model(src, tgt_in).argmax(dim=-1)

        valid = tgt_out != PAD
        ok = (pred == tgt_out) & valid
        tok_ok += ok.sum().item()
        tot_tok += valid.sum().item()
        seq_ok += ((ok | ~valid).all(dim=1)).sum().item()
        tot_seq += tgt_out.shape[0]

    model.train(was_training)
    return tok_ok / tot_tok, seq_ok / tot_seq


def show_examples(model, n=6, length=6):
    """挑几条实际的看看，比任何数字都直观。"""
    torch.manual_seed(123)
    src, tgt_in, tgt_out = gen_batch(n, length, length, BUF)
    src = src.to(device)
    out = model.greedy_decode(src)

    rows = []
    for i in range(n):
        s = [t for t in src[i].tolist() if t < N_DIGITS]
        p = [t for t in out[i, 1:].tolist() if t < N_DIGITS]
        want = [t for t in tgt_out[i].tolist() if t < N_DIGITS]
        rows.append((" ".join(map(str, s)), " ".join(map(str, p)), " ".join(map(str, want))))
    return rows


# ============================================================== 主流程
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-train-len", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos", type=str, default="learned", choices=["learned", "sinusoidal"])
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--tag", type=str, default="reverse")
    args = p.parse_args()

    # 位置编码表长度固定为 POS_LEN，训练和评估共用同一张表。
    # 表里超出训练长度的位置没被更新过，正好用来暴露外推问题。
    max_len_pos = POS_LEN

    print("=" * 72)
    print("阶段 2：序列反转（encoder-decoder）")
    print("=" * 72)
    print(f"  设备         : {device}")
    print(f"  位置编码      : {args.pos}"
          + ("（可学习 embedding）" if args.pos == 'learned' else "（论文的 sin/cos 公式）"))
    print(f"  层数/头数/维度 : {args.n_layer} / {args.n_head} / {args.d_model}")
    print(f"  训练序列长度   : 1 ~ {args.max_train_len}")
    print(f"  训练步数      : {args.max_iters}")
    print()

    torch.manual_seed(0)
    model = Seq2Seq(
        VOCAB, args.d_model, args.n_head, args.n_layer, max_len_pos, args.dropout, args.pos
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量 : {n_params:,} ({n_params / 1e6:.3f}M)")

    # 先看看没训练时是什么样子，作为对照
    tok_acc, seq_acc = evaluate(model, args.max_train_len, n_batches=2)
    print(f"  训练前的整句准确率: {seq_acc * 100:.2f}%  （随机猜，接近 0）")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / (args.warmup + 1)
        r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return 1e-5 + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - 1e-5)

    print("-" * 72)
    print(f"{'步数':>6}  {'loss':>8}  {'token准确':>9}  {'整句准确':>9}  {'lr':>9}  {'耗时':>7}")
    print("-" * 72)

    log = []
    t0 = time.time()

    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % args.eval_interval == 0 or it == args.max_iters:
            tok_acc, seq_acc = evaluate(model, args.max_train_len)
            elapsed = time.time() - t0
            print(f"{it:>6}  {'-':>8}  {tok_acc * 100:>8.1f}%  {seq_acc * 100:>8.1f}%  "
                  f"{lr_at(it):>9.2e}  {elapsed:>6.1f}s")
            log.append((it, seq_acc))
            model.train()

        if it == args.max_iters:
            break

        src, tgt_in, tgt_out = gen_batch(args.batch_size, 1, args.max_train_len, BUF)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        logits = model(src, tgt_in)
        # ignore_index=PAD：填充位置不参与 loss。
        # 这是"变长序列"在训练里的标准处理方式，论文一句话没提，但少了它
        # 模型会花大量精力去学"预测填充符"，白白浪费容量。
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), tgt_out.reshape(-1), ignore_index=PAD
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 100 == 0 and it > 0:
            print(f"{it:>6}  {loss.item():>8.4f}  {'':>9}  {'':>9}  {'':>9}  "
                  f"{time.time() - t0:>6.1f}s")

    total = time.time() - t0
    model.eval()
    print("-" * 72)
    print(f"  训练完成，用时 {total:.1f}s")
    print()

    # ---- 训练长度内 vs 超出训练长度
    print("=" * 72)
    print("泛化检验")
    print("=" * 72)
    print(f"  {'序列长度':<12}{'token准确':>12}{'整句准确':>12}")
    print("  " + "-" * 34)
    for L in [4, 8, 10, 12]:
        t_acc, s_acc = evaluate(model, L, n_batches=4, seed=7)
        mark = "   <- 训练时见过" if L <= args.max_train_len else "   <- 训练时没见过"
        print(f"  {L:<12}{t_acc * 100:>11.1f}%{s_acc * 100:>11.1f}%{mark}")
    print()
    print("  读法：在训练长度内能到 99%，超出长度的掉多少，就说明位置编码的外推能力有多差。")
    if args.pos == "learned":
        print("  用可学习 embedding 时，超出长度的位置压根没被训练过，数值是随机的，所以必然崩。")
        print(f"  想知道正弦编码能不能救，再跑一次：  --pos sinusoidal --tag reverse_sin")
    else:
        print("  正弦编码是写死的公式，任何长度都能算出对应的位置向量，所以理论上不崩。")
        print("  但注意：位置对得上不等于模型会用——注意力权重是学出来的，")
        print("  训练时只见过长度 8 以内的交互模式，长了照样会退化。")
    print()

    # ---- 实例
    print("=" * 72)
    print("生成实例（贪心解码，一条一条吐出来的）")
    print("=" * 72)
    print(f"  {'输入':<22}{'模型输出':<22}{'正确答案':<22}")
    print("  " + "-" * 62)
    for s, pred, want in show_examples(model, n=6, length=6):
        ok = "  OK" if pred == want else "  X"
        print(f"  {s:<22}{pred:<22}{want:<22}{ok}")
    print()
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"seq2seq_{args.tag}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "vocab": {"N_DIGITS": N_DIGITS, "PAD": PAD, "BOS": BOS, "EOS": EOS, "VOCAB": VOCAB},
        },
        ckpt_path,
    )
    csv_path = os.path.join(OUT_DIR, f"acc_{args.tag}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,seq_accuracy\n")
        for it, acc in log:
            f.write(f"{it},{acc:.6f}\n")

    print(f"  权重已保存: {ckpt_path}")
    print(f"  准确率曲线: {csv_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "out")

if __name__ == "__main__":
    main()
