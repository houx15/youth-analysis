# 省级Gender Norm Index构建方法

## 输入

- 各省份的word2vec模型（共31个省份）
- 词表文件：
  - 性别词：male_words, female_words
  - 概念词：family_words, work_words, leadership_words, non_leadership_words, stem_words, non_stem_words

---

## Step 0: OOV检查

在正式计算之前，先检查各词表在各省模型中的覆盖情况。

### 0.1 检查每个词在每个省模型中是否存在

对于每个省份 $p$，每个词表 $L$（gender, family, work, leadership, non_leadership, stem, non_stem），记录：
- 词表总词数 $|L|$
- 在该省模型中找到的词数 $|L_p^{\text{found}}|$
- OOV词列表

### 0.2 输出OOV诊断表

**表1：各省各词表覆盖率**

| province | gender_male | gender_female | family | work | leadership | non_leadership | stem | non_stem |
|----------|-------------|---------------|--------|------|------------|----------------|------|----------|
| 北京 | 30/32 | 35/37 | 32/38 | 33/35 | 20/23 | 16/18 | 18/22 | 19/21 |
| 上海 | 31/32 | 36/37 | 33/38 | 34/35 | 21/23 | 17/18 | 19/22 | 20/21 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**表2：各词的跨省覆盖情况**

| word | category | n_provinces_found | missing_provinces |
|------|----------|-------------------|-------------------|
| 在家带娃 | family | 5 | 北京,上海,... |
| STEM | stem | 12 | ... |
| ... | ... | ... | ... |

### 0.3 决策标准

- 如果某个词在超过50%的省份OOV → 考虑从词表中移除
- 如果某个省份某词表覆盖率低于50% → 该省该维度的index可能不可靠，需标记
- 多词短语（如"在家带娃"、"接送上下学"）预期OOV率较高，提前做好心理准备

### 0.4 处理建议

根据OOV检查结果，决定：
1. 是否需要移除某些高OOV词
2. 是否需要补充某些词表
3. 是否需要对某些省份的结果做标记或排除

---

## Step 1: 构建各省性别轴

对于每个省份 $p$：

1. 提取所有男性词向量，计算质心：
$$\vec{c}_{\text{male},p} = \frac{1}{|M|}\sum_{m \in M} \vec{v}_m$$

2. 提取所有女性词向量，计算质心：
$$\vec{c}_{\text{female},p} = \frac{1}{|F|}\sum_{f \in F} \vec{v}_f$$

3. 构建性别轴并归一化：
$$\vec{g}_p = \frac{\vec{c}_{\text{female},p} - \vec{c}_{\text{male},p}}{||\vec{c}_{\text{female},p} - \vec{c}_{\text{male},p}||}$$

**注意**：记录每个省份有多少性别词在词表中找到（OOV情况）。

---

## Step 2: 计算各概念词在性别轴上的投影

对于每个省份 $p$，每个概念词 $w$：

$$\text{proj}_{w,p} = \vec{v}_w \cdot \vec{g}_p$$

因为 $\vec{g}_p$ 已归一化，这等价于计算余弦相似度乘以词向量的模长。如果你想要纯粹的方向信息，可以用：

$$\text{proj}_{w,p} = \cos(\vec{v}_w, \vec{g}_p) = \frac{\vec{v}_w \cdot \vec{g}_p}{||\vec{v}_w||}$$

**建议使用后者（余弦相似度）**，这样消除了词向量长度的影响。

**输出**：一个表格，columns = [province, word, word_category, projection]

---

## Step 3: 检查跨省可比性

### 3.1 描述性统计

对每个省份，计算所有概念词投影值的：
- 均值 $\mu_p$
- 标准差 $\sigma_p$
- 最小值、最大值

### 3.2 可视化

绘制分省份的boxplot或violin plot：
- x轴：省份
- y轴：投影值（所有概念词）

**判断标准**：
- 如果各省boxplot的位置（中位数）和spread（IQR）大致相似 → 不需要额外标准化
- 如果有明显的系统性差异 → 考虑省内标准化

### 3.3 可选：省内标准化

如果需要标准化，对每个省份的投影值做z-score：

$$z_{w,p} = \frac{\text{proj}_{w,p} - \mu_p}{\sigma_p}$$

其中 $\mu_p, \sigma_p$ 是该省所有概念词投影值的均值和标准差。

---

## Step 4: 计算WEAT效应量

对于每个省份 $p$，每个维度（以work/family为例）：

### 4.1 计算各组均值

$$\bar{X}_{\text{family},p} = \frac{1}{|W_{\text{family}}|}\sum_{w \in W_{\text{family}}} \text{proj}_{w,p}$$

$$\bar{X}_{\text{work},p} = \frac{1}{|W_{\text{work}}|}\sum_{w \in W_{\text{work}}} \text{proj}_{w,p}$$

### 4.2 计算pooled标准差

$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

其中 $s_1, s_2$ 分别是family词和work词投影值的标准差，$n_1, n_2$ 是词数。

### 4.3 计算Cohen's d

$$d_p^{\text{work/family}} = \frac{\bar{X}_{\text{family},p} - \bar{X}_{\text{work},p}}{s_{\text{pooled}}}$$

**解释**：
- $d > 0$：family词更靠近女性端，work词更靠近男性端 → 传统性别规范
- $d < 0$：相反方向
- $|d| \approx 0.2$ 小效应，$|d| \approx 0.5$ 中效应，$|d| \approx 0.8$ 大效应

### 4.4 对leadership和stem维度重复计算

$$d_p^{\text{leadership}} = \frac{\bar{X}_{\text{non-leadership},p} - \bar{X}_{\text{leadership},p}}{s_{\text{pooled}}}$$

$$d_p^{\text{stem}} = \frac{\bar{X}_{\text{non-stem},p} - \bar{X}_{\text{stem},p}}{s_{\text{pooled}}}$$

---

## Step 5: 输出格式

### 5.1 主结果表（用于后续分析）

保存为CSV，格式：

| province | d_work_family | d_leadership | d_stem | n_gender_words | n_family | n_work | n_leadership | n_non_leadership | n_stem | n_non_stem |
|----------|---------------|--------------|--------|----------------|----------|--------|--------------|------------------|--------|------------|
| 北京 | 0.45 | 0.32 | 0.28 | 18 | 12 | 15 | 8 | 10 | 9 | 11 |
| 上海 | 0.38 | 0.29 | 0.31 | 18 | 12 | 14 | 8 | 10 | 9 | 11 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**注**：n_xxx列记录各词表实际找到的词数（非OOV），用于后续稳健性检验。

### 5.2 中间结果表（用于诊断和稳健性检验）

保存所有词的投影值，格式：

| province | word | category | projection | projection_zscore |
|----------|------|----------|------------|-------------------|
| 北京 | 家庭 | family | 0.12 | 0.85 |
| 北京 | 工作 | work | -0.08 | -0.42 |
| ... | ... | ... | ... | ... |

### 5.3 可视化输出（PDF）

1. **跨省可比性诊断图**：各省所有概念词投影值的boxplot
2. **三个index的分布图**：各省效应量的histogram或density plot
3. **地图可视化**（可选）：三个index的省级choropleth map

---

## Step 6: 稳健性检验（后续）

有了上述输出，可以做以下检验：

1. **词表敏感性**：随机drop 20%的词，重新计算index，看相关性
2. **标准化敏感性**：比较标准化前后index的相关性
3. **OOV影响**：检查n_xxx列，看词覆盖率是否和index相关
4. **与调查数据的相关**：convergent validity
5. **与行为数据的相关**：predictive validity

---

## 参考文献

- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183-186.
- Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *PNAS*, 115(16), E3635-E3644.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.