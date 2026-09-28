"""
阶段 3：中文语料 + BPE 分词器 —— 走一遍完整的预训练流程

================================================================================
【四个阶段的关系：把这张表当路线图看】
================================================================================
  阶段 0  step0_attention.py   注意力内部在算什么        一个模块，纯形状演示
  阶段 1  step1_tiny_gpt.py    字符级 GPT               学会"语言的形式"
  阶段 2  step2_seq2seq.py     encoder-decoder          补齐论文的左半边
  阶段 3  step3_bpe_gpt.py     真实分词器 + 中文语料      完整流程，这一步才是工业做法

================================================================================
【这一阶段比阶段 1 多了什么，四件事，每一件都是工业界真正在用的】
================================================================================
  1. BPE 分词器     阶段 1 是字符级（词表 65），这里用 GPT-4 同款（10 万词条）
  2. 中文语料       阶段 1 是 1.1MB 英文，这里是 223 万汉字公版古籍
  3. 混合精度 bf16   显存减半、速度明显变快。Blackwell 原生支持
  4. 梯度累积       显存装不下想要的 batch 时，用小 batch 攒几步再更新一次
                    —— 12GB 显卡训大模型的常规操作，必学
  5. 词表裁剪       用不到的词条直接删掉，输出层从 38.5M 参数砍到 0.4M
                    代价为零，收益是模型小四倍、训练快数倍。见第 1 部分

================================================================================
【和论文的对照】
================================================================================
  论文 3.4 节说 embedding 要乘 sqrt(d_model)。现代实现都不乘了，这里和阶段 1 一样省略。
  论文 5.3 节说 Adam 的 β2 用 0.98、学习率按公式 warmup，这里用 GPT-2 的 β2=0.95，
  因为语料规模比论文小、梯度噪声结构不同。这类"论文这么说、实践那么做"的差异，
  是这个阶段你最该留意的东西。

================================================================================
【运行方式】
================================================================================
  首次运行（会先做数据准备，约 1 分钟，然后开始训练）：
      "C:/Program Files/ComfyUI-aki-v3/python/python.exe" step3_bpe_gpt.py --max-iters 2000

  只测速度不做完整训练：
      "C:/Program Files/ComfyUI-aki-v3/python/python.exe" step3_bpe_gpt.py --max-iters 50

  第二次运行会自动复用已编码好的数据文件，跳过准备步骤。
================================================================================
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出保险

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")


# ==============================================================================
#  第 0 部分：超参数
#      全部集中在这里。想改配置做实验，只改这一段就够了。
# ==============================================================================
class Config:
    # ---- 模型结构 ----
    n_layer = 6            # 堆 6 层 Block
    n_head = 6             # 6 个注意力头
    n_embd = 384           # d_model。384 / 6 = 64，每个头 64 维
    block_size = 512       # 上下文窗口，比阶段 1 的 256 翻倍
    dropout = 0.1

    # ---- 数据 ----
    val_ratio = 0.02       # 2% 留作验证集。中文古籍风格统一，2% 足够

    # ---- 训练 ----
    micro_batch = 8        # 一次真正塞进显卡的批大小（受显存限制的那个）
    grad_accum = 4         # 梯度累积几步再更新一次
    # 【这里是最容易迷糊的地方，仔细看】
    # 有效批大小 = micro_batch × grad_accum × block_size = 8×4×512 = 16384 个 token
    # 为什么要这么绕：显存只装得下 micro_batch=8 这么多样本的中间激活值，
    # 但梯度噪声太大不利于训练，希望"一次更新"看到更多样本。
    # 办法就是攒 4 次梯度再更新一次权重——数学上等价于用 32 的批大小，
    # 显存开销却只有 8 的。这是 12GB 显卡训大模型的标准手段。
    max_iters = 2000
    learning_rate = 6e-4   # GPT-2 的经典取值。阶段 1 用 1e-3 是因为模型更小
    min_lr = 6e-5
    warmup_iters = 200
    weight_decay = 0.1
    beta2 = 0.95           # 论文用的是 0.98，这里跟 GPT-2 用 0.95
    grad_clip = 1.0

    # ---- 日志与评估 ----
    eval_interval = 100
    eval_iters = 50
    seed = 1337


cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")   # 允许 TF32，Blackwell 上矩阵乘法更快


# ==============================================================================
#  第 1 部分：数据准备
#
#  阶段 1 的数据是"读文件 -> 查字符表 -> 完事"。
#  这里多了一步：把文本切成一串 token id，而"怎么切"本身就是一门学问。
# ==============================================================================

def clean_gutenberg(text):
    """
    去掉 Project Gutenberg 的版权头尾。

    为什么要做这一步：这些文件开头有几十行英文法律声明，结尾还有一份
    许可证全文。如果不删，模型会花容量去学这些和法律文书有关的东西——
    而它们和你要它学的中文古籍毫无关系。

    【这是一个通用教训】数据清洗不是可选项。真实项目里，
    清洗代码往往比模型代码还长。
    """
    # Gutenberg 用 *** START OF THE PROJECT GUTENBERG EBOOK XXX *** 标记正文起点
    i = text.find("*** START")
    if i != -1:
        nl = text.find("\n", i)
        text = text[nl + 1:] if nl != -1 else text[i:]

    j = text.find("*** END")
    if j != -1:
        text = text[:j]

    # 去掉 "Produced by ..." 这类制作署名行
    lines = [ln for ln in text.split("\n") if not ln.startswith("Produced by")]

    # 压缩连续空行：古籍排版空行很多，留着会浪费上下文窗口
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def build_dataset(enc, force_rebuild=False):
    """
    把四本书编码成一串 token，存成 .bin 供下次复用。

    【为什么要存成二进制】
    编码 200 多万汉字要几十秒。第一次跑完存下来，之后每次实验直接加载。
    这是工业界的常规做法——真实预训练数据集几百 GB，不可能每次训练都重编码。

    【词表裁剪 —— 本文件里性价比最高的一个优化】
    分词器有 10 万个词条，但你的语料用不到那么多。用不到的词条，
    输出层却要为每一个都算一遍分数（10 万 × 384 维的矩阵乘法），
    既占参数又浪费时间。做法很简单：扫一遍语料，把出现过的词条挑出来
    重新编号成 0,1,2,...，其余的彻底删掉。

    对这套繁体古籍语料，效果极其夸张：100277 -> 1066，只用了 1.1%。
    原因见 main() 里的说明——分词器对繁体古籍几乎退化成字节级编码。
    """
    bin_path = os.path.join(DATA_DIR, "zh_corpus_trimmed.bin")
    vocab_path = os.path.join(DATA_DIR, "zh_corpus_trimmed_vocab.npy")

    if os.path.exists(bin_path) and os.path.exists(vocab_path) and not force_rebuild:
        data = np.fromfile(bin_path, dtype=np.uint32)
        uniq = np.load(vocab_path)
        print(f"  [复用] 已存在编码好的数据: {os.path.basename(bin_path)}")
        print(f"         共 {len(data):,} 个 token，裁剪后词表 {len(uniq):,}")
        return data, uniq

    files = sorted(glob.glob(os.path.join(DATA_DIR, "gutenberg_*.txt")))
    if not files:
        raise FileNotFoundError(f"没有找到语料文件，请确认 {DATA_DIR} 下有 gutenberg_*.txt")

    print(f"  找到 {len(files)} 个语料文件，开始清洗与编码...")
    texts = []
    total_han = 0
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read()
        t = clean_gutenberg(raw)
        n_han = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
        total_han += n_han
        texts.append(t)
        print(f"    {os.path.basename(p):<26} 汉字 {n_han:>9,}  ->  清洗后 {len(t):>9,} 字符")

    full = "\n\n".join(texts)
    print(f"  合计汉字 {total_han:,}，文本 {len(full):,} 字符")

    # ---- 关键的一步：把文本切成 token ----
    # encode_ordinary 的意思是"只按普通规则切，不把文本里的
    # <|endoftext|> 之类特殊标记当成控制符"。处理原始文本时必须用它，
    # 否则文本里万一出现类似字样，会被误解析成特殊 token。
    ids = np.array(enc.encode_ordinary(full), dtype=np.int64)
    print(f"  编码后 {len(ids):,} 个 token")
    print(f"  平均 1 个 token 只覆盖 {len(full) / len(ids):.2f} 个字符（越小说明切得越碎）")

    # ---- 词表裁剪 ----
    # np.unique 得到语料中出现过的所有 token（已排序），
    # lut 是一张查找表：lut[旧编号] = 新编号，没出现过的填 -1。
    # 然后把整串 id 通过 lut 一次性重映射（numpy 花式索引，比 python 循环快几千倍）。
    uniq = np.unique(ids)
    lut = np.full(enc.n_vocab, -1, dtype=np.int64)
    lut[uniq] = np.arange(len(uniq), dtype=np.int64)
    ids = lut[ids].astype(np.uint32)

    print(f"  词表裁剪：{enc.n_vocab:,} -> {len(uniq):,}"
          f"（利用率 {len(uniq) / enc.n_vocab * 100:.1f}%）")

    ids.tofile(bin_path)
    np.save(vocab_path, uniq)
    print(f"  已保存: {bin_path}")
    return ids, uniq


def compare_tokenizers(text_sample):
    """
    【这段一定要看，是阶段 3 最有价值的知识点之一】

    同一段中文，不同的分词器压缩率差很多。这不是小差别，它直接决定：
        同样的 512 长度上下文窗口，到底能装多少内容。

    原因：BPE 是按"字节"起步逐步合并的。
      gpt2 分词器主要用英文语料训练，中文在它的词表里几乎没有整字的词条，
      一个汉字（UTF-8 占 3 字节）会被切成 2~3 个 token。
      cl100k_base（GPT-4 用的）见过大量中文，一个汉字往往只要 1~1.5 个 token。
    """
    import tiktoken

    print("  分词器对比（同一段中文）：")
    print(f"    {'分词器':<16}{'词表大小':>10}{'token 数':>10}{'字/token':>10}")
    print("    " + "-" * 46)
    for name in ["gpt2", "cl100k_base"]:
        try:
            e = tiktoken.get_encoding(name)
        except Exception:
            print(f"    {name:<16}{'不可用':>10}")
            continue
        n = len(e.encode_ordinary(text_sample))
        ratio = len(text_sample) / n
        print(f"    {name:<16}{e.n_vocab:>10,}{n:>10,}{ratio:>9.2f}")
    print()
    print("    最后一列 = 1 个 token 平均覆盖几个字符，越大越省上下文（越高效）。")
    print("    gpt2 的词表只有 5 万，中文被切得很碎；cl100k 有 10 万，中文效率高得多。")
    print("    但这只是「现代简体」的表现——换成繁体古籍，cl100k 会明显退化，")
    print("    因为它的中文词条几乎都是现代简体词，古汉语用字大多不在里面。")
    print()


def get_batch(data, block_size, batch_size, device):
    """
    随机切一批训练样本。

    和阶段 1 完全一样的思路（y 就是 x 右移一格），只有一点不同：
    这里没有"点号"，而是随机抽起点，所以同一个起点可能被抽到多次。
    对 223 万 token 的数据来说这不是问题。真正的预训练会按 epoch 顺序遍历，
    并在多卡之间分片——那是更大规模才需要操心的事。

    整个数据集是"一串连续的 token"，我们只是随机开个窗口：
        x = data[i     : i + block_size    ]
        y = data[i + 1 : i + block_size + 1]
    位置 t 的模型要看 data[i+t]，并预测 data[i+t+1]。
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i: i + block_size] for i in ix])
    y = np.stack([data[i + 1: i + block_size + 1] for i in ix])
    # 转成 int64 是 PyTorch Embedding 的要求（它不接受 uint32）
    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    return x, y


# ==============================================================================
#  第 2 部分：模型
#
#  结构和阶段 1 是同一套，只有三处变化：
#      1. 词表从 65 变成 100277（分词器换了）
#      2. 规模从 2.73M 变成约 49M
#      3. 多了一个生成时的 top-k 采样
#  所以下面这个部分可以对照 step1_tiny_gpt.py 来看，看差异在哪、为什么。
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力。和阶段 0 的手写版本逐字对应，只是封装成了模块。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.w_q = nn.Linear(n_embd, n_embd, bias=False)
        self.w_k = nn.Linear(n_embd, n_embd, bias=False)
        self.w_v = nn.Linear(n_embd, n_embd, bias=False)
        self.w_o = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # 拆头：把最后一维 C 劈成 (n_head, d_head)，再把头维挪到前面
        # 结果形状 (B, n_head, T, d_head)，这样最后两维永远是 (T, d_head)，
        # 矩阵乘法不用关心有几个头。
        q = self.w_q(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 核心一步：Q @ K^T，形状变成 (B, n_head, T, T)
        # 这个 T×T 就是注意力的复杂度 O(T²) 的来源。
        # block_size 从 256 提到 512，这个矩阵的规模翻了 4 倍。
        att = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        # 因果 mask：位置 t 不许看 t 之后的东西，否则等于偷看答案
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))

        att = self.dropout(F.softmax(att, dim=-1))
        out = att @ v                                   # (B, n_head, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)      # 拼回头
        return self.w_o(out)                            # 再过一次输出投影


class FeedForward(nn.Module):
    """论文的 Position-wise FFN：升维 4 倍 -> 激活 -> 降回来。

    论文用 ReLU，现代实现用 GELU。差异很小，但这是"论文和实现不一致"的典型例子。
    参数量上，这一层通常比注意力层还大：
        4 * n_embd²  (注意力四个矩阵)
        8 * n_embd²  (FFN 两个矩阵，中间 4 倍宽)
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
    """一层 = 注意力 + 前馈，各自套残差。

    写成 x = x + f(LayerNorm(x)) 而不是论文的 LayerNorm(x + f(x))，
    原因见阶段 1 的注释：pre-norm 的梯度通路更直，深层更稳。
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


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying，省一份参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 0.02 是 GPT-2 的标准初始化标准差，别改成默认的 1/sqrt(fan_in)，
        # 那个值会让 logits 一开始就很大，softmax 饱和，梯度消失。
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)                              # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)                             # 广播相加
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                             # (B, T, vocab)

        if targets is None:
            return logits, None
        # 摊平成 (B*T, vocab) 对 (B*T,)，让每个位置都参与一次交叉熵。
        # 【注意】这个 logits 张量的形状是 B*T×100277，
        # 在 micro_batch=8、block=512 时是 4096×100277 ≈ 4.1 亿个数，
        # fp32 下就是 1.6 GB —— 这是本脚本里最大的一块显存开销。
        # 想省显存，优先调小 micro_batch，而不是调小模型。
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        """
        自回归生成。比阶段 1 多了一个 top_k，原因值得说清楚：

        语言模型的原始输出分布有个长尾——除了少数几个合理的下一个字，
        后面还挂着几万个概率极低但非零的候选。随机采样时这些小概率事件
        累积起来，就会时不时蹦出一个完全不搭界的字，一句话读下来很脏。

        top_k 的做法：每次只在概率最高的 k 个候选里采样，其余直接砍成负无穷。
        k=200 对 10 万词表来说是很保守的截断，既保留多样性又砍掉噪声。

        temperature 和 top_k 是两件不同的事：
            temperature 改变分布的"尖锐程度"（形状）
            top_k       改变候选的"数量"（范围）
        工业界通常两个一起用。再往上还有 top_p（按累积概率截断），后面可以自己试。
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # 只关心最后一个位置的预测

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 第 k 大的那个值就是门槛，比它还小的全部砍掉
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
#  第 3 部分：训练
# ==============================================================================

def get_lr(it):
    """warmup + 余弦衰减。和阶段 1 一样，只是数值换成 GPT-2 的尺度。"""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """在 train / val 上各采样若干批算平均 loss。

    这两个数的差距就是"过拟合程度"的读数。
    如果 train 一直降而 val 开始往上走，说明模型在背训练数据，该停了。
    """
    was_training = model.training
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg.block_size, cfg.micro_batch, device)
            # 评估时也用 bf16，和训练保持一致，否则 loss 对不上
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train(was_training)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=cfg.max_iters)
    p.add_argument("--micro-batch", type=int, default=cfg.micro_batch)
    p.add_argument("--grad-accum", type=int, default=cfg.grad_accum)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--n-layer", type=int, default=cfg.n_layer)
    p.add_argument("--n-embd", type=int, default=cfg.n_embd)
    p.add_argument("--n-head", type=int, default=cfg.n_head)
    p.add_argument("--tag", type=str, default="zh")
    p.add_argument("--rebuild-data", action="store_true", help="强制重新编码语料")
    p.add_argument("--resume", type=str, default=None,
                   help="加载已训好的权重，跳过训练直接生成（换提示词反复试的时候用）")
    args = p.parse_args()

    cfg.max_iters = args.max_iters
    cfg.micro_batch = args.micro_batch
    cfg.grad_accum = args.grad_accum
    cfg.block_size = args.block_size
    cfg.n_layer = args.n_layer
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 78)
    print("阶段 3：中文语料 + BPE 分词器")
    print("=" * 78)

    # ---------------------------------------------------------------- 分词器
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    print("[1/4] 分词器")
    print("  使用 cl100k_base（GPT-4 同款，总词表 100,277）")
    print("  阶段 1 用的是字符级，词表只有 65 —— 差距在哪，下面这张表会说明")
    print()
    sample = "话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。"
    compare_tokenizers(sample)

    # ---------------------------------------------------------------- 数据
    print("[2/4] 数据")
    data, uniq = build_dataset(enc, force_rebuild=args.rebuild_data)
    vocab_size = len(uniq)
    print()
    print(f"  【看这里】词表利用率只有 {vocab_size / enc.n_vocab * 100:.1f}%，这件事得说清楚：")
    print("  cl100k_base 的中文词条基本是现代简体词，而这套语料是繁体古籍，")
    print("  它大部分不认识，只能退化成字节级来切（1 个汉字 ≈ 1.5 个 token）。")
    print("  于是裁剪后词表只剩一千出头。这不是坏事，反而让模型小了四倍多。")
    print("  换成简体语料利用率会高不少，但依然远达不到 100%。")
    print("  「分词器效率取决于语料分布」这件事，比分词器本身的名字重要得多。")
    print()

    n = int(len(data) * (1 - cfg.val_ratio))
    train_data, val_data = data[:n], data[n:]
    print(f"  训练集 {len(train_data):,} token")
    print(f"  验证集 {len(val_data):,} token")
    eff_batch_tokens = cfg.micro_batch * cfg.grad_accum * cfg.block_size
    epoch_steps = len(train_data) / eff_batch_tokens
    print(f"  有效批大小 {cfg.micro_batch}×{cfg.grad_accum}×{cfg.block_size} = {eff_batch_tokens:,} token/步")
    print(f"  跑 {cfg.max_iters} 步相当于看 {cfg.max_iters / epoch_steps:.1f} 遍完整语料")
    print(f"  看不懂的 baseline：随机猜 loss = ln({vocab_size}) = {math.log(vocab_size):.4f}")
    print()

    # ---------------------------------------------------------------- 模型
    print("[3/4] 模型")
    model = GPT(vocab_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size, cfg.dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    emb_params = model.tok_emb.weight.numel()
    print(f"  结构        {cfg.n_layer} 层 / {cfg.n_head} 头 / d_model={cfg.n_embd} / 上下文 {cfg.block_size}")
    print(f"  总参数      {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"  其中词表    {emb_params:,} ({emb_params / 1e6:.2f}M, 占 {emb_params / n_params * 100:.1f}%)")
    print(f"  实际计算参数 {n_params - emb_params:,} ({(n_params - emb_params) / 1e6:.1f}M)")
    print()
    print("  对比一下裁不裁词表的差别：")
    print("    不裁剪   词表 100,277 -> 输出层 38.5M 参数，占 78%，模型 49.3M")
    print(f"    裁剪后   词表 {vocab_size:>6,} -> 输出层 {emb_params / 1e6:.2f}M 参数，"
          f"占 {emb_params / n_params * 100:.0f}%，模型 {n_params / 1e6:.1f}M")
    print("  表达能力一点没少：被砍掉的那些词条，本来就是永远不会被生成的输出。")
    print()
    print("  【补充】真正的工业预训练不会裁得这么狠（要留泛化余量，")
    print("  不同数据分片的词表也不一样）。这里是封闭语料的教学场景，裁了没损失。")
    print()
    print(f"  设备        {device}")
    if device.type == "cuda":
        print(f"  显卡        {torch.cuda.get_device_name(0)}")
        print(f"  显存        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # --resume：加载已训好的权重，同时把 max_iters 设成 0，
    # 训练循环会立刻退出去、直接走到生成阶段。
    # 用途：换提示词、换温度反复试的时候，不用每次重训七分钟。
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  [跳过训练] 已加载权重: {args.resume}")
        if "config" in ckpt:
            print(f"             该权重的结构: {ckpt['config']['n_layer']} 层 / "
                  f"d_model={ckpt['config']['n_embd']} / 上下文 {ckpt['config']['block_size']}")
            print("             （如果这些数字和上面打印的不一致，说明参数没对上）")
        print()
        cfg.max_iters = 0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        betas=(0.9, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    # ---------------------------------------------------------------- 训练
    print("[4/4] 训练")
    print("-" * 78)
    print(f"{'步数':>6}  {'train':>8}  {'val':>8}  {'ppl':>8}  {'lr':>9}  "
          f"{'token/s':>9}  {'显存':>7}  {'耗时':>7}")
    print("-" * 78)

    log_rows = []
    t0 = time.time()
    t_last = t0
    tokens_seen = 0
    model.train()

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data)
            now = time.time()
            dt = max(now - t_last, 1e-6)
            tps = tokens_seen / dt if tokens_seen else 0
            tokens_seen = 0
            t_last = now
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(f"{it:>6}  {losses['train']:>8.4f}  {losses['val']:>8.4f}  "
                  f"{math.exp(losses['val']):>8.2f}  {lr:>9.2e}  {tps:>9.0f}  "
                  f"{mem:>6.2f}G  {now - t0:>6.1f}s")
            log_rows.append((it, losses["train"], losses["val"]))

        if it == cfg.max_iters:
            break

        # ---------- 梯度累积：这一步是阶段 3 的核心新增 ----------
        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum):
            xb, yb = get_batch(train_data, cfg.block_size, cfg.micro_batch, device)
            # bf16 混合精度：前向/反向用 16 位算，权重更新仍是 32 位。
            # 显存和带宽大约减半，速度明显提升，而且 bf16 的动态范围和 fp32 一样，
            # 不像 fp16 那样容易溢出——所以现代训练基本都用 bf16 而不是 fp16。
            # 注意 autocast 只包住前向，不要包 backward。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, loss = model(xb, yb)
            # 除以 grad_accum：4 次累加后总梯度才和"用 4 倍 batch 一次算"等价。
            # 少了这一句，梯度会放大 4 倍，训练直接发散。
            loss = loss / cfg.grad_accum
            loss.backward()
            tokens_seen += xb.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    total = time.time() - t0
    print("-" * 78)
    print(f"  训练完成，用时 {total:.1f}s  ({cfg.max_iters / total:.2f} 步/秒)")
    print(f"  最终 val loss {log_rows[-1][2]:.4f}   perplexity {math.exp(log_rows[-1][2]):.2f}")
    print()

    # ---------------------------------------------------------------- 存盘
    # 【顺序很重要】先存盘，再生成。
    # 第一版我把生成放在前面，结果提示词写错字形抛了异常，
    # 七分钟训出来的权重跟着一起没了，只能重跑。
    # 教训：训练一结束就先落盘，演示环节再崩也不影响成果。
    if args.resume:
        print("=" * 78)
        print("  --resume 模式：权重没有变化，不重复保存。")
        print()
    else:
        ckpt_path = os.path.join(OUT_DIR, f"gpt_{args.tag}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
                    "n_layer": cfg.n_layer, "block_size": cfg.block_size, "dropout": cfg.dropout,
                },
                "encoding": "cl100k_base",
                "trimmed_vocab": uniq,     # 词表裁剪映射表，解码时必须用它反查
            },
            ckpt_path,
        )
        csv_path = os.path.join(OUT_DIR, f"loss_{args.tag}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,train_loss,val_loss\n")
            for row in log_rows:
                f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f}\n")
        print(f"  权重已保存: {ckpt_path}")
        print(f"  损失已保存: {csv_path}")
        print()

    # ---------------------------------------------------------------- 生成
    print("=" * 78)
    print("生成结果（模型自己写的，未经挑选）")
    print("=" * 78)
    print()
    print("  说明：中文和英文不同，它的字与字之间没有空格，")
    print("  所以判断质量要看「词组是否成词、句子结构是否通顺」，而不是看拼写。")
    print()

    # 词表裁剪过之后，模型的输出是「新编号」，解码前要先换回分词器的原编号。
    lut = np.full(enc.n_vocab, -1, dtype=np.int64)
    lut[uniq] = np.arange(len(uniq), dtype=np.int64)

    def to_new_ids(s):
        """
        把提示词编码成裁剪后的 id；语料里没有的字返回 None。

        【踩过的坑】语料是【繁体】，提示词就必须用繁体。
        第一版我写了简体「却说」，结果直接崩——因为「却」这个简体字
        从没在语料里出现过，编码出来是词表外的 id，查表得到 -1，
        拿去索引 embedding 就越界了。
        繁体对应写法是「卻說」。写提示词前先想清楚语料是什么字形。
        """
        new = lut[np.array(enc.encode_ordinary(s), dtype=np.int64)]
        if (new < 0).any():
            return None
        return new.tolist()

    def to_text(seq):
        return enc.decode([int(uniq[i]) for i in seq])

    # 提示词用繁体，和语料保持一致
    prompts = ["卻說", "話說", "次日", "寶玉"]
    for temp in [0.7, 0.9]:
        for pr in prompts:
            ids = to_new_ids(pr)
            if ids is None:
                print(f"  --- 提示「{pr}」跳过：这几个字没在语料里出现过 ---")
                continue
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=180, temperature=temp, top_k=200)
            text = to_text(out[0].tolist())
            print(f"  --- 提示「{pr}」  temperature={temp} ---")
            print("  " + text.replace("\n", "\n  "))
            print()

    print()
    print("  接下来可以自己做的实验（改参数重跑，用 --tag 区分输出）：")
    print("    --n-layer 10            更深的层数，看 loss 能降到多少")
    print("    --block-size 256        上下文减半，tokens/步变少，训练更快")
    print("    --micro-batch 4         显存不够时调小它，同时把 --grad-accum 加倍")
    print("    --max-iters 4000        训更久，观察 val loss 什么时候开始回升（过拟合）")
    print("=" * 78)


if __name__ == "__main__":
    main()
