"""
阶段 4：怎么测一个训好的语言模型

================================================================================
【先记住一句话】"测"不是一件事，是三个层次。混在一起谈，就会退化成"我觉得还行"。
================================================================================
  第一层 · 看数      loss / perplexity / BPC         客观、可比较、能画曲线
  第二层 · 看基线    比"瞎猜"和"只按词频猜"好多少      回答"到底学到东西没有"
  第三层 · 看输出    生成质量的客观量化                数出重复率，不凭感觉

================================================================================
【本脚本要纠正的第二个误解：distinct-n 不能跨温度比较】
================================================================================
第一版按"distinct-2 最高"来挑采样配置，结果它选了温度 1.1——最差的配置。
为什么？温度越高，随机性越大，重复天然越少。distinct-2 从 0.71（温度 0.5）
一路涨到 0.97（温度 1.1），但文本从"像古文"变成了"乱码"。

**靠调高温度把不重复率刷上去，得到的是更乱的文本，不是更好的文本。**
所以 distinct-n 只能在【同一个温度内】用来比较 top-k。

要跨温度比较，得用一个对温度不那么敏感的指标。本脚本用的是
【三字组命中率】：生成的三字组里有多少在真实语料中出现过。
温度高只让内容更野，不会让它"不像中文"，所以这个数字掉得慢、可比。

选配置的正确规则（脚本里实现了）：
    用【可比的指标】做决策 —— 取三字组命中率最高的
    用【不可比的指标】当硬约束 —— distinct-2 不能明显低于人写的文本
门槛也不拍脑袋：distinct-2 的下限直接按真实语料的 85% 自动折算。

另外，脚本会把真实语料自己也算一遍同样的指标，给出一条参考线。
没有参考线的数字是没有意义的——0.9 到底算好还是差？得看人写的是多少。

最后，脚本会输出【三个候选】而不是唯一答案：最像语料、推荐、最多样。
两个指标互相拉扯，这类模型本来就没有单一最优配置，只有取舍。

前两层十几秒就能跑完，第三层慢一些。真正的评估流程就是这样：
先看数排除明显问题，再看基线确认不是幻觉，最后才花时间品生成质量。

================================================================================
【本脚本要纠正的一个常见误解：perplexity 不能跨分词器比较】
================================================================================
会得到这样一组数字：
    阶段 1（字符级，词表 65）  ：val loss 1.49   perplexity 4.41
    阶段 3（BPE，词表 1066）   ：val loss 2.70   perplexity 14.85

看起来阶段 1 的模型好得多？完全错误。它们的分母不一样：
    perplexity = exp(平均每个 token 的交叉熵)
"每个 token"是多少内容，取决于分词器。阶段 1 一个 token 是一个字符，
阶段 3 一个 token 是 0.68 个字符——切得越碎，同样的内容就用越多 token 表示，
perplexity 天然越大。**两个不同词表的 perplexity 之间没有任何可比性。**

那怎么比？用 bits-per-character（BPC）：
    BPC = loss / ln(2) / (平均每个 token 覆盖几个字符)
它的含义是"编码一个字符需要多少比特"，和词表大小、和分词方式都无关。
换分词器、换模型结构，BPC 都可以直接比。

================================================================================
【关于"测试集"的一个真相】
================================================================================
这个模型的训练用的是随机窗口采样（np.random.randint 抽起点），
也就是说整个语料被反复看过了。真正没被训过的，只有语料最后那 2%
（训练时它叫验证集）。

严格来说，只有"没有参与过任何决策的数据"才算 test set。
如果某个数据被用来决定"该用哪个学习率""该训多少步"，
它就已经不是干净的测试集了。

这个模型只训过一次、没有根据验证集调过任何超参，所以这里
把最后 2% 直接当作 test set 使用。但如果以后反复依据它调参，
就得另切一份真正没碰过的数据出来。

================================================================================
【运行】
================================================================================
    python step4_evaluate.py --ckpt out/gpt_zh1.pt
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from step3_bpe_gpt import GPT, DATA_DIR, OUT_DIR, get_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


# ==============================================================================
#  工具：加载模型与数据
# ==============================================================================

def load_everything(ckpt_path, enc):
    """加载权重、数据和词表映射表。

    注意这里的顺序：权重里的输出维度是【裁剪后】的词表大小，
    所以解码时必须配合 data/zh_corpus_trimmed_vocab.npy 那张映射表，
    把模型的输出 id 换回分词器的原编号。少了这一步就是一堆乱码。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]

    model = GPT(c["vocab_size"], c["n_embd"], c["n_head"], c["n_layer"],
                c["block_size"], c["dropout"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    data_path = os.path.join(DATA_DIR, "zh_corpus_trimmed.bin")
    vocab_path = os.path.join(DATA_DIR, "zh_corpus_trimmed_vocab.npy")
    data = np.fromfile(data_path, dtype=np.uint32)
    uniq = np.load(vocab_path)

    # 反查表：裁剪后的 id -> 分词器原 id
    lut = np.full(enc.n_vocab, -1, dtype=np.int64)
    lut[uniq] = np.arange(len(uniq), dtype=np.int64)
    # 正向表：原 id -> 裁剪后 id（编码提示词用）
    fwd = lut

    return model, c, data, uniq, fwd


# ==============================================================================
#  第一层：看数
# ==============================================================================

@torch.no_grad()
def measure_loss(model, data, block_size, batch_size, n_batches=50):
    """在一段数据上采样若干批，返回平均 loss。

    为什么采样而不是全量：这段数据有 8 万个 token，
    分成 512 长度的窗口会高度重叠，采样 50 批已经足够稳定，
    而且能顺便反映模型在不同位置的表现。
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(data, block_size, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            _, loss = model(xb, yb)
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.std(losses))


def bits_per_character(loss, chars_per_token):
    """
    BPC = loss / ln(2) / (平均每个 token 覆盖几个字符)

    推导一下（不难，值得看）：
        loss 的单位是"nat"，也就是以 e 为底的自然对数。
        换成比特要除以 ln(2)。
        得到的数字是"每个 token 多少比特"。
        再除以"每个 token 覆盖几个字符"，才是"每个字符多少比特"。

    BPC 越低越好，而且和分词器无关——这是唯一能公平比较
    "阶段 1 的字符级模型"和"阶段 3 的 BPE 模型"的指标。
    """
    return loss / math.log(2) / chars_per_token


# ==============================================================================
#  第二层：看基线
# ==============================================================================

def unigram_baseline(train_data, test_data, vocab_size):
    """
    一元模型基线：一个"只会按词频随机抽字"的模型。

    它完全不看上下文，唯一的本事是知道"哪个字常见"。
    做法：统计训练集里每个 token 出现的频率，当作预测概率，
    然后在测试集上算交叉熵。

    这个基线的意义：如果训练出来的模型没比它好多少，
    说明模型其实什么结构都没学到，只是背下了字频。
    对小模型来说这是最现实的"地板线"。
    """
    counts = np.bincount(train_data.astype(np.int64), minlength=vocab_size).astype(np.float64)
    probs = counts / counts.sum()
    logp = np.log(np.maximum(probs, 1e-12))

    test_counts = np.bincount(test_data.astype(np.int64), minlength=vocab_size).astype(np.float64)
    total = test_counts.sum()
    return float(-(test_counts * logp).sum() / total)


# ==============================================================================
#  第三层：看输出（把"生成得好不好"变成可比较的数字）
# ==============================================================================

def distinct_ratio(text, n):
    """
    distinct-n：文本里不重复的 n-gram 占全部 n-gram 的比例。

    1.0 = 一个重复都没有；越低说明复读越严重。
    这是衡量小模型生成质量最实用的一个客观指标——
    中文小模型最常见的毛病就是"说着说着开始复读机"，
    而这个毛病肉眼看几段是发现不了的，必须用数字量出来。

    经验参考（中文）：
        distinct-2 > 0.85  比较健康
        0.7 ~ 0.85         有明显重复
        < 0.7              已经进入复读模式
    """
    if len(text) < n:
        return 1.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams)


def longest_run(text):
    """最长连续相同字符的长度。复读机的另一个特征。"""
    best = cur = 0
    prev = None
    for ch in text:
        cur = cur + 1 if ch == prev else 1
        prev = ch
        best = max(best, cur)
    return best


def out_of_vocab_count(text, train_chars):
    """生成了多少个真实语料里从没出现过的字符。

    这个数字要分两种情况看：
      1) 生成了语料里没有的汉字/标点 —— 理论上不该有，
         因为词表裁剪后模型只能从那 1066 个 token 里挑。
      2) 生成了 U+FFFD（就是显示成 � 的那个字符）—— 字节级分词的副作用。
         cl100k_base 处理繁体字时会退化成「半个汉字的字节片段」，
         一个片段单独解码是乱码，只有前后片段接起来才是完整汉字。
         模型挑出一个片段后，下一个 token 没接上，就留下一个 �。

    第 2 种是真实的质量问题，第 1 种几乎不会出现。
    """
    return sum(1 for c in text if c not in train_chars)


def ngram_hit_rate(text, ref_ngrams, n=3):
    """生成文本里有多少个 n-gram 在真实语料中出现过。

    这是判断「生成的东西像不像人话」最有用的客观指标。
    复读、乱码、语序崩坏都会让这个数字掉下来，
    而它受温度的影响远小于 distinct-n —— 温度高只会让内容更野，
    不会让它「不像中文」。所以在不同温度之间，这个指标是可以比的。

    参考区间（中文小模型）：
        > 0.9    基本都在成词成句
        0.7~0.9  大意通顺，局部生造
        < 0.7    已经在拼字了，不成词
    """
    if len(text) < n:
        return 1.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return sum(1 for g in grams if g in ref_ngrams) / len(grams)


@torch.no_grad()
def generate_texts(model, uniq, enc, prompts, temperature, top_k, n_tokens, n_samples=1):
    """给定提示词，生成一批文本。逐条返回。"""
    model.eval()
    results = []
    for pr in prompts:
        old_ids = np.array(enc.encode_ordinary(pr), dtype=np.int64)
        new_ids = np.full(len(old_ids), -1, dtype=np.int64)
        inv = np.full(enc.n_vocab, -1, dtype=np.int64)
        inv[uniq] = np.arange(len(uniq), dtype=np.int64)
        new_ids = inv[old_ids]
        if (new_ids < 0).any():
            continue          # 提示词里有语料外字符，跳过

        for _ in range(n_samples):
            idx = torch.tensor([new_ids.tolist()], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=n_tokens,
                                 temperature=temperature, top_k=top_k)
            results.append(enc.decode([int(uniq[i]) for i in out[0].tolist()]))
    return results


def quality_report(texts, train_chars, ref_ngrams):
    """把一组生成文本折算成几个客观数字。"""
    if not texts:
        return None
    return {
        "d1": np.mean([distinct_ratio(t, 1) for t in texts]),
        "d2": np.mean([distinct_ratio(t, 2) for t in texts]),
        "d3": np.mean([distinct_ratio(t, 3) for t in texts]),
        "run": np.mean([longest_run(t) for t in texts]),
        "oov": np.mean([out_of_vocab_count(t, train_chars) for t in texts]),
        "hit": np.mean([ngram_hit_rate(t, ref_ngrams, 3) for t in texts]),
        "len": np.mean([len(t) for t in texts]),
    }


# ==============================================================================
#  主流程
# ==============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=os.path.join(OUT_DIR, "gpt_zh1.pt"))
    p.add_argument("--n-samples", type=int, default=3, help="每个温度配置采样几条")
    p.add_argument("--n-tokens", type=int, default=150)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--reuse", action="store_true",
                   help="复用上次的生成缓存，跳过最贵的采样阶段")
    p.add_argument("--cache", type=str, default=None,
                   help="生成缓存路径，默认 out/eval_grid_<权重名>.json")
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    print("=" * 78)
    print("阶段 4：评估一个训好的语言模型")
    print("=" * 78)
    print()

    model, c, data, uniq, fwd = load_everything(args.ckpt, enc)
    vocab_size = c["vocab_size"]
    block_size = c["block_size"]

    print(f"  权重     {args.ckpt}")
    print(f"  结构     {c['n_layer']} 层 / {c['n_head']} 头 / d_model={c['n_embd']} / 上下文 {block_size}")
    print(f"  词表     {vocab_size:,}（裁剪后）")
    print(f"  参数     {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ---- 数据切分 ----
    # 训练时用了前 98%，最后 2% 从未参与梯度更新，这里当作 test set。
    n = int(len(data) * 0.98)
    train_data, test_data = data[:n], data[n:]
    print(f"  数据     训练段 {len(train_data):,} token / 测试段 {len(test_data):,} token")
    print(f"           测试段和训练时用的验证集是同一段——详见本文件开头的说明")
    print()

    # ================================================================ 第一层
    print("=" * 78)
    print("第一层 · 看数")
    print("=" * 78)

    t0 = time.time()
    train_loss, train_std = measure_loss(model, train_data, block_size, 8, args.eval_batches)
    test_loss, test_std = measure_loss(model, test_data, block_size, 8, args.eval_batches)

    # 平均每个 token 覆盖多少字符 —— 用整段语料算
    # 4,157,445 个 token 对应 2,830,016 个字符
    chars_per_token = 2830016 / 4157445

    print(f"  训练段 loss    {train_loss:.4f}  (±{train_std:.4f})")
    print(f"  测试段 loss    {test_loss:.4f}  (±{test_std:.4f})")
    print(f"  差距          {test_loss - train_loss:+.4f}")
    print()
    print("  这个差距就是过拟合的读数。参考判断：")
    print("    < 0.3   健康，模型还没开始背训练数据")
    print("    0.3~0.6 轻度过拟合，还能再训一会儿")
    print("    > 0.6   明显在背了，该加大 dropout 或加数据")
    print()
    print(f"  测试段 perplexity  = exp({test_loss:.4f}) = {math.exp(test_loss):.2f}")
    print(f"  测试段 BPC         = {bits_per_character(test_loss, chars_per_token):.4f} 比特/字符")
    print()
    print("  【这就是那个能跨分词器比较的指标】")
    print("  perplexity 只在同一套词表内部有意义，BPC 才和分词方式无关。")
    print("  换分词器、换模型结构时，拿 BPC 对比就行，perplexity 别拿来比。")
    print()

    # ================================================================ 第二层
    print("=" * 78)
    print("第二层 · 看基线（回答：模型到底学到东西没有）")
    print("=" * 78)

    uniform = math.log(vocab_size)
    unigram = unigram_baseline(train_data, test_data, vocab_size)

    print(f"  {'方案':<28}{'测试段 loss':>12}{'BPC':>10}{'相对改进了':>12}")
    print("  " + "-" * 62)

    def row(name, loss):
        imp = (uniform - loss) / uniform * 100
        print(f"  {name:<28}{loss:>12.4f}{bits_per_character(loss, chars_per_token):>10.4f}"
              f"{imp:>11.1f}%")

    row("均匀瞎猜（什么都不学）", uniform)
    row("只按字频猜（一元模型）", unigram)
    row("本实验模型", test_loss)
    print()
    print("  最后一列 = 相对'均匀瞎猜'减少了多少不确定性。")
    print(f"  从瞎猜到只知道字频，就干掉了 {(uniform - unigram) / uniform * 100:.1f}%——")
    print("  这一部分几乎不需要任何「智能」，只要统计一下词频。")
    print(f"  从字频到本实验模型，又干掉了 {(unigram - test_loss) / unigram * 100:.1f}%，")
    print("  这一部分是上下文带来的——也就是注意力真正在做的事。")
    print()

    # ================================================================ 第三层
    print("=" * 78)
    print("第三层 · 看输出（把'生成得好不好'变成数字）")
    print("=" * 78)
    print()

    # ---- 把真实语料解码成文本，作为「人写的文本长什么样」的参考线 ----
    # 【关键 1】必须整段解码，不能一个 token 一个 token 地解码。
    #   字节级片段单独解码只会得到一堆 �，整段解码才能还原成正确汉字。
    # 【关键 2】参考表要建在【整段语料】上，不能只取一小段。
    #   三字组命中率的分母是"参考表里有没有这个三字组"，
    #   参考表越小，命中率被低估得越厉害，会冤枉模型。
    # 分段解码只是为了别一次构造 400 万个 Python 整数，结果拼接起来是一样的。
    CHUNK = 500_000
    parts = []
    for s in range(0, len(data), CHUNK):
        parts.append(enc.decode([int(v) for v in uniq[data[s:s + CHUNK]]]))
    train_text = "".join(parts)
    train_chars = set(train_text)
    ref_ngrams = set(train_text[i:i + 3] for i in range(len(train_text) - 2))
    print(f"  参考语料：整段解码 {len(data):,} 个 token → {len(train_text):,} 个字符")
    print(f"            {len(train_chars):,} 个不同字符，{len(ref_ngrams):,} 个不同三字组")
    print()

    prompts = ["卻說", "話說", "次日", "寶玉"]
    temps = [0.5, 0.7, 0.9, 1.1]
    topks = [50, 200, 1000]

    print("  温度 × top-k 网格，每格采样 "
          f"{args.n_samples} 条 × {len(prompts)} 个提示词，各 {args.n_tokens} 字")
    print()

    # 真实语料自己也过一遍同样的指标 —— 有了这条参考线，下面的数字才有意义
    # 注意：它的 3字组命中必然是 1.000，因为参考表就是从语料建的（属于定义使然）。
    # 真正有意义的是 distinct-2 和最长复读：那是"人写的中文长什么样"的实测值。
    ref_sample = [train_text[i:i + args.n_tokens] for i in range(0, 20000, 5000)]
    ref_rep = quality_report(ref_sample, train_chars, ref_ngrams)

    def gprint(temp, k, rep):
        print(f"  {f'{temp} / {k}':>10}  {rep['len']:>5.0f}  {rep['d1']:>11.3f}  "
              f"{rep['d2']:>11.3f}  {rep['run']:>9.1f}  {rep['oov']:>9.1f}  "
              f"{rep['hit']:>10.3f}")

    print(f"  {'配置':>10}  {'长':>5}  {'distinct-1':>11}  {'distinct-2':>11}  "
          f"{'最长复读':>9}  {'语料外字':>9}  {'3字组命中':>10}")
    print("  " + "-" * 82)
    print(f"  {'真实语料':>10}  {ref_rep['len']:>5.0f}  {ref_rep['d1']:>11.3f}  "
          f"{ref_rep['d2']:>11.3f}  {ref_rep['run']:>9.1f}  {ref_rep['oov']:>9.1f}  "
          f"{ref_rep['hit']:>10.3f}  ← 人写的，参考线")
    print("  " + "-" * 82)

    # 生成是整个脚本最贵的一步（每格 12 条 × 150 字，而且没有 KV cache，
    # 每吐一个字都要把整条前缀重算一遍）。所以把结果缓存下来，
    # 下次改输出格式、改分析方法时用 --reuse 直接读，不用重跑十几分钟。
    cache_path = args.cache or os.path.join(
        OUT_DIR, "eval_grid_" + os.path.basename(args.ckpt).replace(".pt", "") + ".json")

    rows = []
    if args.reuse and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for item in json.load(f):
                rows.append((item["temp"], item["topk"], item["rep"], item["texts"]))
        for temp in temps:
            for k in topks:
                hit = [r for r in rows if r[0] == temp and r[1] == k]
                if hit:
                    gprint(*hit[0][:3])
            print()
        print(f"  【本次直接读缓存】{cache_path}")
        print("  想真跑一遍新的采样配置，去掉 --reuse 即可。")
        print()
    else:
        for temp in temps:
            for k in topks:
                texts = generate_texts(model, uniq, enc, prompts, temp, k,
                                       args.n_tokens, args.n_samples)
                rep = quality_report(texts, train_chars, ref_ngrams)
                if rep is None:
                    continue
                rows.append((temp, k, rep, texts))
                gprint(temp, k, rep)
            print()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([{"temp": t, "topk": k, "rep": rp, "texts": tx}
                       for t, k, rp, tx in rows], f, ensure_ascii=False)
        print(f"  生成结果已缓存 → {cache_path}")
        print("  下次加 --reuse 可省掉这十几分钟（只改分析、不换模型时用）。")
        print()

    print("  怎么读这张表：")
    print("    distinct-2 越高越好，< 0.7 说明进入复读模式。")
    print("               但它【不能跨温度比较】：温度越高随机性越大、重复天然越少，")
    print("               靠调高温度把 distinct-2 刷到 0.96，得到的不是更好的文本，")
    print("               而是更乱的文本。只在【同一个温度内】比 top-k，才是合法的。")
    print("    最长复读   越小越好，正常中文里连续相同字符不该超过 3")
    print("    3字组命中  生成的三字组有多少在真实语料里出现过。这个指标可以跨温度用，")
    print("               因为温度只改变内容的野不野，不改变它像不像中文。")
    print("               但要注意：**复读会让它虚高**（重复的三字组天然在语料里），")
    print("               所以它必须和 distinct-2 配着看，单独用会被骗。")
    print("    语料外字   生成了多少个语料里没有的字符，主要是 �（字节片段拼不上）")
    print()

    # ---- 怎么选配置：一个「可比指标」做决策 + 一个「硬约束」把关 ----
    # 关键认识：这两个指标互相拉扯，温度把它们拉向相反方向。
    #   3字组命中 ↑ = 更像语料（正确性）      —— 可以跨温度比
    #   distinct-2 ↑ = 更多样（但太高就是乱）  —— 不可以跨温度比
    # 所以正确做法是：
    #   用【可比的指标】做决策：取命中率最高
    #   用【不可比的指标】当硬约束：不能复读、不能乱
    # distinct-2 的门槛不拍脑袋，直接按真实语料打折算出来，自动自适应。
    bar_d2 = 0.85 * ref_rep["d2"]
    bar_run = 3.0
    pool = [r for r in rows if r[2]["d2"] >= bar_d2 and r[2]["run"] <= bar_run]
    if not pool:
        pool = [max(rows, key=lambda r: r[2]["d2"])]

    by_hit = max(rows, key=lambda r: r[2]["hit"])            # 最像语料
    by_bal = max(pool, key=lambda r: r[2]["hit"])            # 满足约束里命中最高
    by_div = max(rows, key=lambda r: r[2]["d2"])             # 最多样（大概率是坑）

    print("=" * 78)
    print("候选方案：评估的正确产出不是一个数字，是一条取舍曲线")
    print("=" * 78)
    print(f"  {'候选':<11}{'温度':>5}{'top-k':>7}{'3字组命中':>11}{'distinct-2':>12}"
          f"{'最长复读':>10}{'语料外字':>10}")
    print("  " + "-" * 70)
    print(f"  {'真实语料':<11}{'—':>5}{'—':>7}{ref_rep['hit']:>11.3f}"
          f"{ref_rep['d2']:>12.3f}{ref_rep['run']:>10.1f}{ref_rep['oov']:>10.1f}"
          f"  ← 人写的，参考线")

    def crow(name, r, tag=""):
        t_, k_, rp, _ = r
        print(f"  {name:<11}{t_:>5}{k_:>7}{rp['hit']:>11.3f}{rp['d2']:>12.3f}"
              f"{rp['run']:>10.1f}{rp['oov']:>10.1f}  {tag}")

    crow("命中最高", by_hit, "← 单一指标选的坏答案")
    crow("推荐", by_bal, "← 命中与不重复兼顾")
    crow("最多样", by_div, "← 另一个坏答案")
    print()
    print(f"  distinct-2 门槛 = 真实语料的 85% = {bar_d2:.3f}"
          "（按参考线自动折算，不拍脑袋）")
    print()
    print("  【这一节是整份脚本最值钱的地方】")
    print("  上面两个候选都是「被单一指标选出来的坏答案」，但坏的方向正好相反：")
    print(f"    命中最高那个：命中 {by_hit[2]['hit']:.3f} 是全场第一，"
          f"但 distinct-2 只有 {by_hit[2]['d2']:.3f}——")
    print("      它在复读（把同一个词反复吐，比如「魚魚魚魚魚」）。")
    print("      复读的文本三字组高度重复，而重复的三字组恰好都被语料覆盖，")
    print("      于是命中率被刷得虚高。**命中率会被复读骗到。**")
    print(f"    最多样那个：distinct-2 {by_div[2]['d2']:.3f} 是全场第一，"
          f"但命中只有 {by_div[2]['hit']:.3f}——")
    print("      它已经不成词了。用高温度把重复率刷下去，得到的是更乱的文本。")
    print("      **distinct-2 会被乱码骗到。**")
    print()
    print("  所以任何单一指标都能被某种退化的输出骗到。")
    print("  正确做法是「一个决策指标 + 一个硬约束」的组合，缺一个就会选错：")
    print("      决策用命中率（可跨温度比），约束用 distinct-2（不许复读）。")
    print()

    temp, k, rep, texts = by_bal

    print("=" * 78)
    print(f"推荐配置：温度 {temp}，top-k {k}")
    print("=" * 78)
    print()

    for label, r in (("推荐", by_bal), ("命中最高（对照）", by_hit)):
        t_, k_, rp, tx = r
        print(f"  —— {label}：温度 {t_} / top-k {k_}，"
              f"命中 {rp['hit']:.3f} / distinct-2 {rp['d2']:.3f} ——")
        for s in tx[:2]:
            print("    " + s.replace("\n", "\n    "))
        print()

    # ================================================================ 总结
    print("=" * 78)
    print("总结：怎么向别人说明「这个模型行不行」")
    print("=" * 78)
    print()
    print(f"  1. 测试段 loss {test_loss:.4f}，BPC {bits_per_character(test_loss, chars_per_token):.4f} 比特/字符")
    print(f"  2. 比只按字频猜的基线改进了 {(unigram - test_loss) / unigram * 100:.1f}%")
    print(f"  3. 训练/测试 loss 差距 {test_loss - train_loss:+.4f}，过拟合程度可接受")
    print(f"  4. 推荐采样配置：温度 {temp}，top-k {k}，三字组命中 {rep['hit']:.3f}")
    print(f"     （真实语料自己是 {ref_rep['hit']:.3f} —— 这就是还差的量）")
    print()
    print("  这四条摆出来，比「我觉得生成的还挺像」有说服力得多。")
    print("  评估的意义不是证明模型好，而是说明它在什么条件下会变差。")
    print()


if __name__ == "__main__":
    main()
