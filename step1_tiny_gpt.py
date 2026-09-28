"""
阶段 1：字符级迷你 GPT —— 让 Transformer 真的学起来

阶段 0 验证了注意力模块的形状。本脚本把它组装成一个完整的、
能训练的 GPT，然后在莎士比亚全集上训练，看它学会写"像英语的东西"。

规模说明（为什么不用论文那个配置）：
    论文 base 版：6 层 / 8 头 / d_model=512 / 6 千万参数 / WMT14 450 万句对 / 8 卡 12 小时
    这里：        6 层 / 6 头 / d_model=192 / 270 万参数 / 1.1MB 纯文本 / 单卡几分钟
    架构是同一个，只是把规模砍到能在单机上几分钟跑完。
    换来的好处：一小时内就能看到完整的"训练 -> loss 下降 -> 生成文本"闭环。

运行（先用小步数试通）：
    python step1_tiny_gpt.py --max-iters 100
完整训练：
    python step1_tiny_gpt.py
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

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "tinyshakespeare.txt")
OUT_DIR = os.path.join(ROOT, "out")


# ============================================================== 超参数
# 全部集中在这里，改完再跑一次就能看差别。
class Config:
    batch_size = 64        # 一次并行处理几段文本
    block_size = 256       # 每段文本多长（= 上下文窗口 = 注意力的 T）
    n_embd = 192           # d_model
    n_head = 6             # 注意力头数
    n_layer = 6            # 堆几层 Block
    dropout = 0.1
    learning_rate = 1e-3   # 小模型可以用大一点的学习率
    min_lr = 1e-4
    warmup_iters = 200     # 论文里的 warmup：前 200 步学习率从 0 线性升上来
    max_iters = 3000
    eval_interval = 250
    eval_iters = 50        # 每次评估用多少批算平均，越大越稳越慢
    grad_clip = 1.0        # 梯度裁剪：防止偶发的梯度爆炸把模型踹飞
    seed = 1337


cfg = Config()


# ============================================================== 数据
def load_data(path):
    """
    字符级 tokenizer：词表就是文本里出现过的所有字符。

    为什么不直接用 BPE / tiktoken：字符级没有"分词"这一步，
    "一个 token = 一个字符"，全流程透明，每一个数字从哪来都能看清。
    代价是序列更长、效果更差——但学习阶段这不是缺点。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 编号
    itos = {i: ch for i, ch in enumerate(chars)}   # 编号 -> 字符

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, stoi, itos, text


def get_batch(split, train_data, val_data, vocab_size):
    """
    取一批训练数据。这是整个训练循环里最容易被忽略、但最关键的一步：

        x = "To be or not to b"
        y = "o be or not to be"     <- y 就是 x 右移一位

    也就是说，模型在位置 i 看到的输入是 token[i]，要求它预测 token[i+1]。
    这就是"自监督"：数据本身既是输入也是标签，不需要人工标注。
    整个 GPT 的训练目标只有这一件事，做完这件事，语言能力就长出来了。
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i: i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1: i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================== 模型
class MultiHeadAttention(nn.Module):
    """和阶段 0 第 8 步完全一致的模块，只是把 mask 换成随长度动态生成。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.w_q = nn.Linear(n_embd, n_embd, bias=False)
        self.w_k = nn.Linear(n_embd, n_embd, bias=False)
        self.w_v = nn.Linear(n_embd, n_embd, bias=False)
        self.w_o = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.w_q(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        # 因果 mask：注册成 buffer，跟着模型走（存 checkpoint 时会一起保存）
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.w_o(out)


class FeedForward(nn.Module):
    """
    论文里的 Position-wise Feed-Forward Network。

    结构：Linear(d -> 4d) -> 激活 -> Linear(4d -> d)
    这就是论文图里那个方框，4 行代码。

    两个细节和论文不同，说明一下：
    1. 论文用 ReLU，现代实现（GPT-2 起）用 GELU。差异很小，GELU 更平滑。
    2. 中间层放大 4 倍是论文的设定，一直沿用到现在。
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    一个完整层 = 注意力 + 前馈，每块都套残差。

    注意这里和论文的一个差异：
        论文（post-norm）：x = LayerNorm(x + Sublayer(x))
        这里（pre-norm） ：x = x + Sublayer(LayerNorm(x))

    为什么改：post-norm 在层数多的时候训练不稳定，需要精细调 warmup。
    pre-norm 让残差通路上的梯度是一条"高速公路"，深层也能训得动。
    现在几乎所有大模型都用 pre-norm，翻开源码看到的都是这个写法。

    残差为什么重要（一句话）：x + f(x) 求导时那个 1 保证梯度能原样传回去，
    不会每过一层就衰减一次。没有残差，6 层就够呛，更别说 100 层。
    """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        # 两层 embedding
        self.tok_emb = nn.Embedding(vocab_size, n_embd)      # 词表里的编号 -> 向量
        self.pos_emb = nn.Embedding(block_size, n_embd)      # 位置编号 -> 向量

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 权重共享（论文里的 weight tying）：
        # 输入查表和输出打分用的是同一个矩阵的转置。
        # 好处：省一份参数，而且实验发现效果更好（输入输出在同一语义空间里）。
        # 论文里还乘了个 sqrt(d_model)，现代实现基本都省略了。
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.tok_emb(idx)                     # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        # 广播相加：(B,T,C) + (T,C) -> (B,T,C)
        #
        # 为什么必须有位置编码：注意力算的是"任意两个位置的内积"，
        # 它天生对顺序无感——把输入序列打乱，输出只是跟着换个顺序，信息量不变。
        # 但语言是有顺序的（"狗咬人" != "人咬狗"）。所以必须把位置信息加进去。
        # 论文用的是固定的正弦函数，这里用可学习的位置 embedding——更简单，
        # 效果一样好，而且省掉了那个 sin/cos 公式。
        x = self.drop(tok + pos)

        x = self.blocks(x)                          # 6 层，每层形状不变
        x = self.ln_f(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # 交叉熵。注意必须先摊平：
        #   logits:  (B, T, 65) -> (B*T, 65)
        #   targets: (B, T)     -> (B*T,)
        # 这一步是"每个位置都要预测下一个字符"的代码体现——
        # 不是只在句尾预测一次，而是 T 个位置同时各自预测，一次梯度更新用上 T 倍的信号。
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """
        自回归生成：每次只预测一个字符，把这个字符接到输入后面，再预测下一个。

        注意 idx[:, -self.block_size:]：上下文不能超过 block_size，
        超了就只保留最近的 block_size 个字符。这就是"上下文窗口"的由来。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature    # 只取最后一个位置的预测
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================== 训练
def get_lr(it):
    """论文的 warmup + 余弦衰减。"""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, vocab_size):
    """在 train / val 上各算一批平均 loss，用来判断是真的在学还是死记硬背。"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, train_data, val_data, vocab_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=cfg.max_iters)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--n-layer", type=int, default=cfg.n_layer)
    p.add_argument("--n-embd", type=int, default=cfg.n_embd)
    p.add_argument("--n-head", type=int, default=cfg.n_head)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--tag", type=str, default="default", help="输出文件后缀，方便对比不同配置")
    args = p.parse_args()

    cfg.max_iters = args.max_iters
    cfg.batch_size = args.batch_size
    cfg.n_layer = args.n_layer
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head
    cfg.block_size = args.block_size
    cfg.seed = args.seed

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("字符级迷你 GPT —— 训练")
    print("=" * 70)
    print(f"  设备        : {device}")
    if device.type == "cuda":
        print(f"  显卡        : {torch.cuda.get_device_name(0)}")
    print(f"  层数/头数/维度: {cfg.n_layer} / {cfg.n_head} / {cfg.n_embd}")
    print(f"  上下文长度   : {cfg.block_size}")
    print(f"  批大小      : {cfg.batch_size}")
    print(f"  训练步数     : {cfg.max_iters}")
    print()

    # ---- 数据
    train_data, val_data, vocab_size, stoi, itos, text = load_data(DATA_PATH)
    print(f"  语料        : {len(text):,} 字符")
    print(f"  词表大小     : {vocab_size}")
    print(f"  字符集       : {''.join(repr(c)[1:-1] for c in sorted(set(text)))[:70]} ...")
    print(f"  训练/验证    : {len(train_data):,} / {len(val_data):,}")
    print()
    print(f"  看不懂的 baseline：随机猜的 loss 是 ln({vocab_size}) = {math.log(vocab_size):.4f}")
    print("  训练的目标就是把这个数往下压。压到 1.5 以下，生成的文本就开始能看了。")
    print()

    # ---- 模型
    torch.manual_seed(cfg.seed)
    model = TinyGPT(vocab_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size, cfg.dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量   : {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  权重占用     : {n_params * 4 / 1e6:.1f} MB (fp32)")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )

    # ---- 训练循环
    print("-" * 70)
    print(f"{'步数':>6}  {'train':>8}  {'val':>8}  {'ppl':>7}  {'lr':>9}  {'耗时':>7}  进度")
    print("-" * 70)

    log_rows = []
    t0 = time.time()
    best_val = float("inf")

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data, vocab_size)
            ppl = math.exp(losses["val"])
            elapsed = time.time() - t0
            frac = max(0.0, min(1.0, 1.0 - losses["val"] / math.log(vocab_size)))
            bar = "#" * int(28 * frac) + "." * (28 - int(28 * frac))
            print(
                f"{it:>6}  {losses['train']:>8.4f}  {losses['val']:>8.4f}  "
                f"{ppl:>7.2f}  {lr:>9.2e}  {elapsed:>6.1f}s  {bar}"
            )
            log_rows.append((it, losses["train"], losses["val"]))
            best_val = min(best_val, losses["val"])

        if it == cfg.max_iters:
            break

        xb, yb = get_batch("train", train_data, val_data, vocab_size)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：把所有梯度的整体范数限制在 grad_clip 以内。
        # 不加它，偶尔一个异常 batch 会产生巨大梯度，一步就把训练好的权重毁掉。
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    total = time.time() - t0
    print("-" * 70)
    print(f"  训练完成，用时 {total:.1f}s  ({cfg.max_iters / total:.1f} 步/秒)")
    print(f"  最终 val loss = {log_rows[-1][2]:.4f}   perplexity = {math.exp(log_rows[-1][2]):.2f}")
    print()

    # ---- 生成
    print("=" * 70)
    print("生成结果（模型自己写的，未经任何挑选）")
    print("=" * 70)
    model.eval()
    for temp in [0.5, 0.8, 1.0]:
        start = torch.tensor([[stoi["\n"]]], dtype=torch.long, device=device)
        out = model.generate(start, max_new_tokens=400, temperature=temp)
        gen_text = "".join(itos[i] for i in out[0].tolist())
        print()
        print(f"--- temperature = {temp} ---")
        print(gen_text)
    print()
    print("=" * 70)
    print("temperature 在干什么：")
    print("  它不改变模型，只是在采样前把 logits 除以这个数。")
    print("  0.5  -> 分布变尖，只挑最有把握的字符，文本保守但容易重复")
    print("  1.0  -> 原始分布，更发散，也更容易出现拼写崩坏")
    print("  模型越训得好，能承受的 temperature 越高。上面三段对比就是这个意思。")

    # ---- 存盘
    ckpt_path = os.path.join(OUT_DIR, f"tiny_gpt_{args.tag}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": cfg.n_embd,
                "n_head": cfg.n_head,
                "n_layer": cfg.n_layer,
                "block_size": cfg.block_size,
                "dropout": cfg.dropout,
            },
            "stoi": stoi,
            "itos": itos,
        },
        ckpt_path,
    )
    csv_path = os.path.join(OUT_DIR, f"loss_{args.tag}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss\n")
        for row in log_rows:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f}\n")

    print()
    print(f"  权重已保存: {ckpt_path}")
    print(f"  损失已保存: {csv_path}")
    print()
    print("  想再跑一次不同配置：改 --n-layer / --n-embd / --max-iters 就行，")
    print("  用 --tag 区分输出文件名，两次的 loss 曲线可以直接叠一起比。")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 允许 TF32：Blackwell 上矩阵乘法能快一截，精度损失在这个任务里看不出来
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    main()
