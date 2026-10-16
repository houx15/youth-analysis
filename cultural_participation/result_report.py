"""读取服务器汇总包，复核结果并生成日期前缀PDF、表格和阶段报告。

本地仅处理小型汇总表，不重新拟合用户模型。
python -m cultural_participation.result_report --date 20261016
"""

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

DOMAINS = ['politics', 'business', 'technology', 'sports', 'arts']
LABELS = ['Politics / governance', 'Economy / business', 'Science / technology', 'Sports', 'Arts / entertainment']
ZH = dict(zip(DOMAINS, ['政治／治理', '经济／商业', '科学／科技', '体育', '文艺／娱乐']))
METRICS = ['retweet_entry', 'expression_entry', 'retweet_share', 'expression_share', 'log_delay', 'comment_on_retweet_share']
TITLES = ['Ever retweeted in domain', 'Ever expressed in domain', 'Share of all retweets', 'Share of all expressive posts']
# 沿用gender_domain/figures.py：性别配色、标记、字体、日期命名和细网格。
COLORS = {'m': '#20AEE6', 'f': '#ff7333'}
MARKERS = {'m': 'o', 'f': '^'}
plt.rcParams.update({'font.sans-serif': ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'pdf.fonttype': 42, 'font.size': 9})


def load(path):
    names = ['descriptives', 'gender_gaps', 'cross_domain_contrasts', 'coverage', 'user_exclusions']
    with tarfile.open(path) as archive:
        tables = {n: pd.read_csv(archive.extractfile(n + '.csv')) for n in names}
        manifest = json.load(archive.extractfile('manifest.json'))
    if manifest.get('status') != 'complete':
        raise ValueError('输入运行未完成')
    sources = manifest['source_manifests']
    if len(sources) != 12 or {s['signature']['year'] for s in sources} != {2020}:
        raise ValueError('需要2020年的12个月份状态清单')
    if {s['signature']['month'] for s in sources} != set(range(1, 13)):
        raise ValueError('月份清单重复或遗漏')
    shared = [{k:v for k,v in s['signature'].items() if k not in ['month','inputs']} for s in sources]
    if any(s['status'] != 'complete' for s in sources) or any(s != shared[0] for s in shared):
        raise ValueError('月份版本不一致或未完成')
    d = tables['descriptives']
    expected = set(itertools.product(DOMAINS, ['m','f'], METRICS))
    if d.duplicated(['domain','gender','metric']).any() or set(zip(d.domain,d.gender,d.metric)) != expected:
        raise ValueError('描述统计有缺项或重复')
    g = tables['gender_gaps']; c = tables['cross_domain_contrasts']
    for table, paired in [(g,False),(c,True)]:
        keys = ['domain','metric','model'] + (['contrast_domain'] if paired else [])
        if table.duplicated(keys).any() or len(table) != (120 if paired else 60):
            raise ValueError('模型表有缺项或重复')
        if not table.status.eq('ok').all() or not np.isfinite(table[['estimate_f_minus_m','se','ci_low','ci_high']]).all().all():
            raise ValueError('存在未成功估计；须明确处理，不能静默绘图')
    for _, row in g[g.model.eq('unadjusted_OLS')].iterrows():
        means = d[(d.domain == row.domain)&(d.metric == row.metric)].set_index('gender')['mean']
        np.testing.assert_allclose(row.estimate_f_minus_m,means.f-means.m,atol=1e-10)
    # 份额的跨域分析使用同一用户集合，应等于单域性别系数之差。
    for _,row in c[c.metric.isin(METRICS[:4])].iterrows():
        effects=g[(g.metric==row.metric)&(g.model==row.model)].set_index('domain').estimate_f_minus_m
        np.testing.assert_allclose(row.estimate_f_minus_m,effects[row.domain]-effects[row.contrast_domain],atol=1e-9)
    return tables,manifest


def style(ax, labels=LABELS):
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.25, linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def points(ax, data, column, scale, offset, color, marker, label):
    x=data[column].to_numpy()*scale
    errors=np.vstack([data[column]-data.ci_low,data.ci_high-data[column]])*scale
    ax.errorbar(x,np.arange(len(data))+offset,xerr=errors,fmt=marker,color=color,
                markersize=4,capsize=2,elinewidth=.9,label=label)


def figure_descriptives(d):
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.2))
    for ax,metric,title in zip(axes.flat,METRICS[:4],TITLES):
        for gender,offset in [('m',-.11),('f',.11)]:
            rows=d[(d.metric==metric)&(d.gender==gender)].set_index('domain').loc[DOMAINS]
            points(ax,rows,'mean',100,offset,COLORS[gender],MARKERS[gender],{'m':'Male','f':'Female'}[gender])
        style(ax);ax.set_title(title,fontsize=10);ax.set_xlabel('Users (%)' if metric.endswith('entry') else 'Mean user share (%)')
        ax.set_xlim(left=0);ax.legend(fontsize=8)
    fig.suptitle('2020 | Gendered participation across five domains',fontsize=12)
    fig.text(.5,.015,'User-weighted means and 95% CIs. Entry: all users; shares: users with nonzero respective denominators. Domains may overlap.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.045,1,.95));return fig


def figure_gaps(g, secondary=False):
    metrics=METRICS[4:] if secondary else METRICS[:4]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3)) if secondary else plt.subplots(2,2,figsize=(11.5,7.2))
    titles=['Retweet delay: mean log(1 + seconds)','Same-domain expression on a retweet'] if secondary else TITLES
    for ax,metric,title in zip(axes.flat,metrics,titles):
        for model,offset,color,marker,label in [('unadjusted_OLS',-.11,'#999999','o','Gender only'),('activity_adjusted_OLS',.11,'#000000','s','+ Activity')]:
            rows=g[(g.metric==metric)&(g.model==model)].set_index('domain').loc[DOMAINS]
            points(ax,rows,'estimate_f_minus_m',1 if metric=='log_delay' else 100,offset,color,marker,label)
        style(ax);ax.axvline(0,color='#777777',lw=.7);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Female minus male (log units)' if metric=='log_delay' else 'Female minus male (percentage points)');ax.legend(fontsize=8)
    fig.suptitle('2020 | Gender gaps: '+('timing and retweet expression' if secondary else 'entry and participation shares'),fontsize=12)
    note='OLS with HC1 95% CIs; activity: log(1 + posts / retweets / active days). Separate-domain eligible samples; descriptive associations.'
    fig.text(.5,.015,note,ha='center',fontsize=7);fig.tight_layout(rect=(0,.065 if secondary else .045,1,.94));return fig


def figure_contrasts(c):
    pairs=list(itertools.combinations(DOMAINS,2))
    fig,axes=plt.subplots(1,2,figsize=(12.5,6))
    labels=[f'{LABELS[DOMAINS.index(a)]} − {LABELS[DOMAINS.index(b)]}' for a,b in pairs]
    for ax,metric,title in zip(axes,['retweet_share','expression_share'],['Retweet share','Expression share']):
        source=c[(c.metric==metric)&(c.model=='activity_adjusted_OLS')].set_index(['domain','contrast_domain'])
        selected=[]
        for a,b in pairs:
            row=source.loc[(a,b)].copy() if (a,b) in source.index else source.loc[(b,a)].copy()
            if (a,b) not in source.index:
                row['estimate_f_minus_m']=-row.estimate_f_minus_m
                row['ci_low'],row['ci_high']=-row.ci_high,-row.ci_low
            selected.append(row)
        rows=pd.DataFrame(selected)
        points(ax,rows,'estimate_f_minus_m',100,0,'#000000','s','+ Activity')
        style(ax,labels);ax.axvline(0,color='#777777',lw=.7);ax.set_title(title);ax.set_xlabel('Difference in female–male gaps (percentage points)')
    fig.suptitle('2020 | Cross-domain differences in gender gaps',fontsize=12)
    fig.text(.5,.02,'OLS on each user’s A − B share, with activity covariates; HC1 pointwise 95% CIs. Positive: more female-skewed in A than B.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.05,1,.94));return fig


def md_table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in r)+'|' for r in rows])


def report(t,m,prefix,archive_hash):
    d=t['descriptives'].set_index(['domain','gender','metric']);g=t['gender_gaps'];c=t['cross_domain_contrasts'];cov=t['coverage'].set_index('gender')
    def mean(domain,gender,metric): return d.loc[(domain,gender,metric),'mean']
    levels=[];effects=[];entry=[];secondary=[]
    for domain in DOMAINS:
        levels.append([ZH[domain]]+[f'{100*mean(domain,sex,metric):.2f}%' for metric in ['retweet_share','expression_share'] for sex in ['m','f']])
        row=g[(g.domain==domain)&(g.model=='activity_adjusted_OLS')].set_index('metric')
        effects.append([ZH[domain]]+[f'{100*row.loc[metric,"estimate_f_minus_m"]:+.2f} [{100*row.loc[metric,"ci_low"]:+.2f}, {100*row.loc[metric,"ci_high"]:+.2f}]' for metric in METRICS[:4]])
        entry.append([ZH[domain]]+[f'{100*mean(domain,sex,metric):.2f}%' for metric in METRICS[:2] for sex in ['m','f']])
        delay=row.loc['log_delay'];comment=row.loc['comment_on_retweet_share']
        secondary.append([ZH[domain],f'{delay.estimate_f_minus_m:+.4f}',f'{100*np.expm1(delay.estimate_f_minus_m):+.2f}%',f'{100*mean(domain,"m","comment_on_retweet_share"):.2f}%',f'{100*mean(domain,"f","comment_on_retweet_share"):.2f}%',f'{100*comment.estimate_f_minus_m:+.2f}'])
    pairrows=[]
    for _,r in c[(c.domain=='arts')&(c.model=='activity_adjusted_OLS')&c.metric.isin(['retweet_share','expression_share'])].iterrows():
        pairrows.append([f'文艺／娱乐 − {ZH[r.contrast_domain]}',{'retweet_share':'转发份额','expression_share':'表达份额'}[r.metric],f'{100*r.estimate_f_minus_m:+.2f}',f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]'])
    activity=[]
    for sex in ['m','f']:
        r=cov.loc[sex];activity.append([{'m':'男性','f':'女性'}[sex],f'{int(r.n_users):,}',f'{r.total_posts/r.n_users:.1f}',f'{r.total_retweets/r.n_users:.1f}',f'{r.active_days/r.n_users:.1f}',f'{100*r.matched_retweets/r.total_retweets:.2f}%',f'{100*r.matched_expression/r.total_expression:.2f}%'])
    return f'''# 五领域文化参与：行为分析阶段报告

日期：{prefix[:4]}-{prefix[4:6]}-{prefix[6:]}。数据年份：2020。输入：服务器已完成的 summary_review.tar.gz。

## 1. 本轮结果与分析对象

观测用户的参与重心具有清楚的领域差异：男性的政治、商业、科技、体育参与份额更高，女性的文艺／娱乐份额更高。控制活动量后方向保持一致。仅用“是否至少参与过一次”会得到不同的图景：原始转发进入率在五个领域中都是女性更高，加入活动量后其中四个领域翻转。

这支持区分进入、参与配置和表达实践。行为差异目前不构成声望、地位评价或贬值的直接证据。

本轮直接分析服务器汇总结果，没有重新拟合用户模型，也没有新增画像控制、账户排除或重采样模型。完整运行清单的12个月份版本一致；输入303个文件、188,258,567行，去重后144,934,465帖。性别一致且已知的225,339名用户纳入分析；6名性别缺失用户被排除，无已记录的性别冲突用户。按既有可观测日期解释，不因日期覆盖问题推迟本轮分析。

{md_table(['性别','用户数','人均发帖','人均转发','人均活跃天数','全部转发命中率','全部表达命中率'],activity)}

最后两列是帖子加权的覆盖诊断，不是下面用户等权的领域均值。女性用户占样本约77%，这里的性别均值和差异均按用户计算，不比较男女帖子总数。机构账号尚未单独识别或排除。

## 2. 指标与模型口径

- 转发进入：至少一次转发原文命中该领域；表达进入：至少一次自身清理后的文本命中该领域。分母为全部分析用户（225,339人）。
- 转发份额：用户领域转发数／用户全部转发数，再按性别取均值。仅包括有转发的216,595名用户（男48,656，女167,939）。原文缺失的转发仍进入分母。
- 表达份额：用户该领域有效表达帖数／用户全部有效表达帖数，再按性别取均值。包括224,285名用户（男51,495，女172,790）。表达可以来自原创或转发附言；清理移除转发链、URL及占位语。
- 转发时同领域表达：领域转发中，自己的附言也命中同一领域的比例。它不是“任意附言”的比例，更不是态度、立场或论证质量。
- 时间差：原帖发布时间至转发时间的正间隔，先对每条取 log(1+秒)，再在用户内取均值。不能解释为用户看到内容后的反应速度。
- 领域允许重叠；五领域份额不要求合计100%，也不是五领域内部归一化份额。
- 模型：用户等权OLS；进入指标使用线性概率模型。调整模型加入 log(1+发帖量)、log(1+转发量)、log(1+活跃天数)。所有系数为女性减男性，95%区间使用HC1标准误。活动量调整是描述性条件比较，不是因果识别。

## 3. 参与份额：五领域的性别模式

{md_table(['领域','男：转发份额','女：转发份额','男：表达份额','女：表达份额'],levels)}

政治／治理的转发份额男性约为女性的2.24倍，表达份额约为5.16倍。文艺／娱乐方向相反，女性转发份额约为男性的1.42倍，表达份额约为1.33倍。倍数描述相应指标内的相对差异，不宜据此直接断言表达的性别效应强于转发：两个指标的分母、基准水平及样本略有不同。

活动量调整后的差异（百分点，方括号为95%置信区间）：

{md_table(['领域','转发进入','表达进入','转发份额','表达份额'],effects)}

份额差异经活动量调整后方向未变。政治转发差距从−6.30降至−5.78个百分点，文艺从+7.54降至+6.93个百分点；不能把这种衰减解释成活动量的因果解释比例。商业表达进入的区间包含0，但商业表达份额的性别差异明确：相近的进入率可以与不同的投入份额并存。

## 4. 进入与参与配置：为什么需要多个指标

{md_table(['领域','男：转发进入','女：转发进入','男：表达进入','女：表达进入'],entry)}

女性人均转发约449次，男性约396次；全年是否至少发生一次参与会随行为机会增多而上升。加入活动量变量后，女性转发进入差异在政治、商业、科技、体育分别为−5.29、−2.65、−4.96、−4.63个百分点，文艺仍为+1.62个百分点。这与活动量有关的解释一致，但模型翻转本身不是机制验证。

研究叙述可以据此从“谁参与哪个领域”细化为“广泛进入之下，谁把更多参与份额配置给哪个领域”。不能把高进入率写成没有领域分化，也不能把低份额写成完全不参与。

## 5. 跨领域直接比较

每位用户先计算领域A份额减领域B份额，再以性别和活动量回归；因此保留了同一用户不同领域之间的相关性，而非把两个独立标准误相加。

{md_table(['领域对比','指标','女性−男性的跨域差异（百分点）','95%区间'],pairrows)}

文艺与其他四个领域的份额性别差异都得到直接对比支持。政治与文艺的调整后转发差异相差12.71个百分点，表达差异相差6.49个百分点；这是两种份额指标各自的跨域对比，不应互相当作同尺度强度比较。

其他领域也存在内部梯度。政治比商业、科技、体育更偏男性；科技与体育的调整后转发份额性别差异仅相差约0.09个百分点，区间跨0。由此可以把“五领域差异”理解为有梯度的配置模式，不能把四个领域都视为完全同质。图中的区间为逐项95%区间，尚未进行多重比较校正；接近零的细微差异不作为主要结论。

## 6. 转发时间与转发时表达

{md_table(['领域','调整后时间差系数','exp(系数)−1','男：同域附言比例','女：同域附言比例','调整后附言差异（百分点）'],secondary)}

前四个领域的女性时间差系数为正，文艺为负。exp(系数)−1表示对用户平均log(1+秒)这一结果的指数变换，不能写成算术平均等待时间增加了相同比例。文艺的效应约−2.82%，相对较小。不同领域模型纳入各自有正时间差命中的用户，跨领域时间模型则只纳入两边都可计算的用户，不能直接相减单域系数替代配对模型。

转发时同领域表达的调整后差异在四个非文艺领域都约−2个百分点，文艺为+1.13个百分点。这个结果与份额的方向一致，但词表再次命中限制了指标的含义。例如“说得对”这类附言不会被识别为同域表达。四个非文艺领域之间，该指标的调整后配对差异区间全部包含0；现阶段没有证据为它们排出稳定的细分顺序。

## 7. 当前可以形成的论文结果线索

1. 进入：男女普遍参与多个领域，原始转发进入率还受到行为量差异影响。
2. 配置：参与重心沿领域发生性别分化；这种模式在活动量条件比较后仍存在。
3. 表达：进入相近并不意味着表达份额相近；领域内表达与转发呈现相关但不相同的性别模式。
4. 领域结构：文艺与其余领域存在稳定反向模式，政治差异尤其突出，而科技和体育在转发份额上接近。

本轮可支撑的经验命题是“社交媒体文化参与的领域配置及表达实践存在性别分化”。文化声望／地位及性别关联需要独立测量后再与行为连接，不能由这五个领域的结果自行推出贬值。年龄、教育、地区等画像因素，以及机构账号构成，尚不在当前调整模型中。

## 8. 产物、核验与复现

- [图集PDF]({prefix}_cultural_participation_figures.pdf)：四页，依次为原始水平、调整前后差异、时间与附言、跨领域份额对比。
- 单图：{prefix}_01_levels.pdf、{prefix}_02_gaps.pdf、{prefix}_03_timing_expression.pdf、{prefix}_04_cross_domain.pdf。
- [分析核验清单]({prefix}_analysis_audit.json)：输入哈希、版本、样本量与核验项目。
- 复现：`python3 -m cultural_participation.result_report --date {prefix}`，读取本地outputs中的小型汇总压缩包；不运行原始语料扫描。
- 生成代码：`cultural_participation/result_report.py`。图使用旧流程的男蓝#20AEE6、女橙#ff7333、圆点／三角标记、细网格和PDF日期前缀；系数图用灰色圆点和黑色方块区分模型，避免把模型颜色误认作性别。
- 输入SHA256：`{archive_hash}`。
- 原始处理版本：`{m['source_manifests'][0]['signature']['git_sha']}`。

核验包括月份清单完整性及共同版本、描述与模型表格结构、模型成功状态、原始女性减男性系数与均值之差一致、共同样本的跨领域系数与单域系数差一致。原始数据、词表和账号识别的测量有效性不由这些数值一致性检查替代。
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='cultural_participation/outputs/summary_review.tar.gz')
    parser.add_argument('--output',default='cultural_participation/reports')
    parser.add_argument('--date',default=datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d'))
    args=parser.parse_args();datetime.strptime(args.date,'%Y%m%d')
    path=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tables,manifest=load(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    # 下文含针对本轮结果撰写的解释；新数据须重新审阅，防止沿用旧结论。
    if digest != '78ba0c0af9dbc1b5d90f0fa1efd7b77e7195600c607cede27cd9e8d2031fd871':
        raise ValueError('输入已改变，请更新本轮报告解释及哈希后再生成')
    figures=[('01_levels',figure_descriptives(tables['descriptives'])),('02_gaps',figure_gaps(tables['gender_gaps'])),
             ('03_timing_expression',figure_gaps(tables['gender_gaps'],True)),('04_cross_domain',figure_contrasts(tables['cross_domain_contrasts']))]
    with PdfPages(out/f'{args.date}_cultural_participation_figures.pdf') as pdf:
        for name,fig in figures:
            fig.savefig(out/f'{args.date}_{name}.pdf',bbox_inches='tight',dpi=300)
            pdf.savefig(fig,bbox_inches='tight',dpi=300)
            fig.savefig(out/f'{args.date}_{name}.png',bbox_inches='tight',dpi=160)
            plt.close(fig)
    (out/f'{args.date}_behavior_analysis.md').write_text(report(tables,manifest,args.date,digest),encoding='utf-8')
    audit={'input_sha256':digest,'source_code_sha':manifest['source_manifests'][0]['signature']['git_sha'],
           'vocabulary_sha256':manifest['source_manifests'][0]['signature']['vocabulary_sha256'],
           'unique_posts':manifest['unique_posts'],'analysis_users':int(tables['coverage'].n_users.sum()),
           'checks':['12 month signatures consistent','60 descriptive cells','60 single-domain models and 120 contrasts successful',
                     'unadjusted coefficients equal gender mean differences','entry/share pairwise coefficients equal domain coefficient differences'],
           'new_models_fitted':False,'ci':'HC1 pointwise 95%, not multiplicity adjusted','date':args.date}
    (out/f'{args.date}_analysis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    for name,table in tables.items():
        table.to_csv(out/f'{args.date}_{name}.csv',index=False)
    print(out/f'{args.date}_behavior_analysis.md')


if __name__=='__main__':
    main()
