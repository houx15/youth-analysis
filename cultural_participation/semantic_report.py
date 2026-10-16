"""全词表语义地图、领域分布、模型比较和稳健性图；日期前缀PDF。"""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from cultural_participation.term_semantics import correlation

DOMAINS=['politics','business','technology','sports','arts']
LABELS=['Politics','Business','Technology','Sports','Arts']
D_COLORS=['#4c72b0','#55a868','#8172b3','#c4ad66','#ff7333']
AXES=['gender','prestige','status','cultivation','potency','morality','affluence']
A_LABELS=['Gender','Esteem (candidate)','Status hierarchy','Cultivation','Potency','Morality','Affluence']
MODELS=['sgns_verified','fasttext_verified']
M_LABELS=['SGNS Mixed-large','fastText CC+Wiki']
plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':9})
CMAP=LinearSegmentedColormap.from_list('blue_orange',['#20AEE6','#ffffff','#ff7333'])


def clean(ax):
    ax.grid(alpha=.2,linewidth=.5);ax.set_axisbelow(True)


def comparisons(tables,root):
    a,b=[tables[m]['term_scores'] for m in MODELS]
    common=a.merge(b,on='term',suffixes=('_sgns','_fasttext'))
    shared=common[common.covered_sgns&common.covered_fasttext].copy()
    rows=[]
    for axis in AXES:
        x,y=shared[axis+'_sgns'],shared[axis+'_fasttext']
        rows.append({'axis':axis,'common_terms':len(shared),'pearson':correlation(x,y),'spearman':correlation(x,y,True)})
    pd.DataFrame(rows).to_csv(root/'model_agreement.csv',index=False)
    ass=[]
    for model in MODELS:
        t=tables[model]['term_scores'];t=t[t.term.isin(shared.term)]
        for axis in AXES[1:]:
            ass.append({'model':model,'evaluation':axis,'n':len(t),'pearson_gender_evaluation':correlation(t.gender,t[axis]),'spearman_gender_evaluation':correlation(t.gender,t[axis],True)})
    pd.DataFrame(ass).to_csv(root/'common_sample_associations.csv',index=False)
    sensitivity=[]
    for model in MODELS:
        t=tables[model]['term_scores'].set_index('term');loo=tables[model]['leave_pair_out_scores']
        for (axis,pair),s in loo.groupby(['axis','removed_pair']):
            original=t.loc[s.term,axis].to_numpy();alternative=s.score.to_numpy()
            sensitivity.append({'model':model,'axis':axis,'variant':'remove '+pair,'rank_agreement':correlation(original,alternative,True),
                                'gender_prestige_r':correlation(alternative,t.loc[s.term,'prestige'].to_numpy()) if axis=='gender' else correlation(t.loc[s.term,'gender'].to_numpy(),alternative) if axis=='prestige' else np.nan})
        for axis in ['gender_kinship','gender_pronoun']:
            s=t.dropna(subset=[axis,'prestige','gender'])
            sensitivity.append({'model':model,'axis':'gender','variant':axis,'rank_agreement':correlation(s.gender,s[axis],True),
                                'gender_prestige_r':correlation(s[axis],s.prestige)})
    pd.DataFrame(sensitivity).to_csv(root/'anchor_sensitivity.csv',index=False)
    return shared,pd.DataFrame(rows),pd.DataFrame(ass),pd.DataFrame(sensitivity)


def coverage_fig(tables):
    fig,ax=plt.subplots(figsize=(9,4.4));y=np.arange(6)
    for i,(model,label) in enumerate(zip(MODELS,M_LABELS)):
        c=tables[model]['coverage'].set_index('domain').loc[['all',*DOMAINS]]
        ax.barh(y+(i-.5)*.32,c.coverage*100,height=.3,color=['#999999','#333333'][i],label=label)
        for j,(_,r) in enumerate(c.iterrows()):ax.text(r.coverage*100+1,j+(i-.5)*.32,f'{r.n_covered:.0f}/{r.n_terms:.0f}',va='center',fontsize=7)
    ax.set_yticks(y,['All terms',*LABELS]);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xlabel('Exact vocabulary coverage (%)');ax.legend(loc='lower left');clean(ax)
    fig.suptitle('Vocabulary coverage | All 3,972 terms retained in the audit',fontsize=12)
    fig.text(.5,.015,'No synthetic subword vectors or missing-score imputation. Multi-domain terms appear in each applicable domain.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.94));return fig


def map_fig(tables):
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharex=True,sharey=True)
    for ax,model,label in zip(axes,MODELS,M_LABELS):
        t=tables[model]['term_scores'];t=t[t.covered]
        for domain,lab,color in zip(DOMAINS,LABELS,D_COLORS):
            sub=t[t.domains.str.split('|').map(lambda x:domain in x)]
            ax.scatter(sub.gender,sub.prestige,s=8,alpha=.22,c=color,label=lab,rasterized=True)
            ax.scatter(sub.gender.mean(),sub.prestige.mean(),s=85,c=color,edgecolor='black',marker='D',linewidth=.7)
        for term in ['人工智能','篮球','足球','电影','综艺','法律']:
            sub=t[t.term==term]
            if len(sub):
                r=sub.iloc[0];ax.annotate(term,(r.gender,r.prestige),xytext=(4,5),textcoords='offset points',fontsize=7)
        ax.axhline(0,color='#999999',lw=.6);ax.axvline(0,color='#999999',lw=.6);clean(ax)
        ax.set_title(f'{label} | n={len(t):,}');ax.set_xlabel('Gender projection: male ← 0 → female')
    axes[0].set_ylabel('Esteem projection: contempt ← 0 → respect');axes[1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Word-level cultural map | Candidate esteem axis',fontsize=12)
    fig.text(.5,.015,'Diamonds: mean word projections. Exact axes, not PCA/UMAP; zeros are axis midpoints, not validated social neutrality. Chinese anchors are provisional.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.045,1,.94));return fig


def domains_fig(tables):
    fig,axes=plt.subplots(2,3,figsize=(12,7.4),sharey=True)
    for i,(model,label) in enumerate(zip(MODELS,M_LABELS)):
        t=tables[model]['term_scores'];t=t[t.covered]
        for j,axis in enumerate(['gender','prestige','status']):
            ax=axes[i,j];values=[t.loc[t.domains.str.split('|').map(lambda x:d in x),axis].dropna().to_numpy() for d in DOMAINS]
            box=ax.boxplot(values,vert=False,patch_artist=True,showfliers=False,widths=.55)
            for patch,color in zip(box['boxes'],D_COLORS):patch.set_facecolor(color);patch.set_alpha(.6)
            ax.set_yticks(range(1,6),LABELS);ax.set_ylim(5.5,.5);ax.axvline(0,c='#777777',lw=.6);clean(ax)
            ax.set_title(label+'\n'+A_LABELS[AXES.index(axis)],fontsize=9);ax.set_xlabel('Projection')
    fig.suptitle('Domain distributions | Variation within domains remains visible',fontsize=12)
    fig.text(.5,.015,'Median, IQR, and 1.5×IQR whiskers; outlier points hidden only in this distribution figure, retained in analysis.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.94));return fig


def robustness_fig(tables):
    variants=['all_covered','exclude_known_person','exclude_organization_rule','exclude_known_brand','exclude_gender_marked','exclude_axis_anchors','exclude_all_flagged_entities','merge_reviewed_aliases']
    labels=['All covered','Drop tagged persons','Drop organization-rule terms','Drop listed brands/platforms','Drop 男/女 terms','Drop axis anchors','Drop all flagged entities','Merge 3 reviewed alias groups']
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
    for ax,model,label in zip(axes,MODELS,M_LABELS):
        a=tables[model]['associations'];a=a[a.scope=='all'].pivot(index='variant',columns='axis',values='pearson').loc[variants,AXES[1:]]
        image=ax.imshow(a,vmin=-.6,vmax=.6,cmap=CMAP,aspect='auto')
        for i in range(len(a)):
            for j in range(len(a.columns)):ax.text(j,i,f'{a.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(6),A_LABELS[1:],rotation=35,ha='right');ax.set_yticks(range(8),labels);ax.set_title(label)
    fig.subplots_adjust(left=.22,right=.91,bottom=.22,top=.85,wspace=.12)
    cax=fig.add_axes([.93,.25,.015,.55]);fig.colorbar(image,cax=cax,label='Pearson r with female association')
    fig.suptitle('Does female association imply lower symbolic evaluation?',fontsize=12)
    fig.text(.5,.02,'Exclusions use explicitly documented tags/rules, not exhaustive entity typing. Colors encode correlation signs; no causal devaluation claim.',ha='center',fontsize=7)
    return fig


def agreement_fig(shared,agreement):
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,axis in zip(axes,['gender','prestige','status']):
        x=shared[axis+'_sgns'].rank(pct=True);y=shared[axis+'_fasttext'].rank(pct=True)
        ax.hexbin(x,y,gridsize=30,mincnt=1,cmap='Greys',linewidths=0)
        ax.plot([0,1],[0,1],color='#ff7333',lw=1)
        r=agreement.set_index('axis').loc[axis,'spearman']
        ax.set_title(f'{A_LABELS[AXES.index(axis)]}\nSpearman r = {r:.3f}');ax.set_xlabel('SGNS rank percentile');clean(ax)
    axes[0].set_ylabel('fastText rank percentile')
    fig.suptitle(f'Cross-model agreement | Identical {len(shared):,}-term sample',fontsize=12)
    fig.text(.5,.02,'Ranks compared on the common sample; model cosine scales are not assumed interchangeable.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.055,1,.93));return fig


def anchors_fig(sensitivity):
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,model,label in zip(axes,MODELS,M_LABELS):
        s=sensitivity[(sensitivity.model==model)&sensitivity.axis.isin(['gender','prestige'])].dropna(subset=['gender_prestige_r'])
        labels=[r.axis+': '+r.variant for _,r in s.iterrows()]
        ax.scatter(s.gender_prestige_r,np.arange(len(s)),c='#333333',s=25)
        ax.set_yticks(range(len(s)),labels,fontsize=7);ax.invert_yaxis();ax.axvline(0,c='#999999',lw=.7);clean(ax);ax.set_title(label);ax.set_xlabel('Gender–esteem Pearson correlation')
    fig.suptitle('Anchor sensitivity | Same words, alternative axis definitions',fontsize=12)
    fig.text(.5,.015,'Each omission removes one entire matched pair. Pronoun/kinship versions measure different gender cues, not interchangeable gold standards.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.05,1,.94));return fig


def md(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='cultural_participation/outputs/semantic_analysis/20261016');p.add_argument('--output',default='cultural_participation/reports');p.add_argument('--date',default='20261016');a=p.parse_args()
    root=Path(a.output)/f'{a.date}_semantic_tables';root.mkdir(parents=True,exist_ok=True)
    tables={model:{name:pd.read_csv(Path(a.input)/model/(name+'.csv')) for name in ['term_scores','coverage','associations','centroids','leave_pair_out_scores','axis_coverage','anchor_alignment','axis_cosines','extremes']} for model in MODELS}
    for model in MODELS:
        for name in ['term_scores','coverage','associations','centroids','axis_coverage','anchor_alignment','axis_cosines','extremes']:
            tables[model][name].to_csv(root/f'{model}_{name}.csv',index=False)
        shutil.copy(Path(a.input)/model/'manifest.json',root/f'{model}_manifest.json')
    shared,agreement,ass,sensitivity=comparisons(tables,root)
    figs=[('01_coverage',coverage_fig(tables)),('02_map',map_fig(tables)),('03_distributions',domains_fig(tables)),
          ('04_robustness',robustness_fig(tables)),('05_model_agreement',agreement_fig(shared,agreement)),('06_anchor_sensitivity',anchors_fig(sensitivity))]
    out=Path(a.output)
    with PdfPages(out/f'{a.date}_semantic_figures.pdf') as pdf:
        for name,fig in figs:
            fig.savefig(out/f'{a.date}_semantic_{name}.pdf',bbox_inches='tight',dpi=300)
            fig.savefig(out/f'{a.date}_semantic_{name}.png',bbox_inches='tight',dpi=150)
            pdf.savefig(fig,bbox_inches='tight',dpi=300);plt.close(fig)
    rows=[]
    for axis in AXES[1:]:
        r=[]
        for model in MODELS:
            source=tables[model]['associations'];r.append(source[(source.variant=='all_covered')&(source.scope=='all')&(source.axis==axis)].iloc[0].pearson)
        rows.append([A_LABELS[AXES.index(axis)],*[f'{x:+.3f}' for x in r]])
    commonrows=[[r.model,r.evaluation,int(r.n),f'{r.pearson_gender_evaluation:+.3f}'] for _,r in ass.iterrows()]
    agreementrows=[[r.axis,int(r.common_terms),f'{r.spearman:.3f}'] for _,r in agreement.iterrows()]
    sensitivity_rows=[[M_LABELS[MODELS.index(r.model)],r.variant,f'{r.gender_prestige_r:+.3f}'] for _,r in sensitivity[sensitivity.variant.isin(['gender_kinship','gender_pronoun'])].iterrows()]
    text=f'''# 全词表性别关联与象征评价：首轮语义分析

日期：{a.date}。阶段：两套静态词向量测量、覆盖诊断、领域质心／分布和稳健性比较完成；词项级行为联结尚待服务器重聚合。

## 1. 本轮实际做了什么

正式词表全部3,972词纳入审计，不因人名、领域或预期性别差异预先筛选。SGNS Mixed-large精确覆盖3,247词（81.75%），fastText CC+Wiki精确覆盖2,844词（71.60%），共同样本2,576词（64.85%）。未登录词保留在表中并标记缺失，不用子词合成、不以零填补。向量只流式提取所需词，未训练新模型或调用现代embedding API。

资源：[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors/blob/master/README_zh.md)、[fastText官方模型](https://fasttext.cc/docs/en/crawl-vectors.html)。模型来自不同语料与训练方法，不能把差异归因于某一个单独因素，也不能把它们都当作2020年社会共识。

## 2. 首轮结果：没有得到统一的女性化贬抑模式

下表为词项的女性关联与各评价轴得分的Pearson相关，每个模型使用自身全部有向量词。正值表示更女性关联的词在该评价轴上更高。它不是因果效应，也不是行为性别差。

{md(['评价轴','SGNS Mixed-large','fastText CC+Wiki'],rows)}

社会尊崇候选轴在SGNS中呈正相关，在fastText中接近零；社会地位轴没有呈现稳定负相关。力量轴在两模型中均为小幅负相关。文化修养与道德评价总体为正，但程度随模型变化。现在不应写成“女性化文化被普遍贬低”，也不应反过来宣布不存在文化贬抑：当前中文轴尚未独立验证，尤其尊崇与地位不是同一个构念。

## 3. 共同词样本与模型稳定性

{md(['模型','评价轴','共同词数','性别—评价相关'],commonrows)}

同一批词在两模型中的排序一致性：

{md(['轴','共同词数','Spearman相关'],agreementrows)}

性别排序的跨模型一致性中等；尊崇候选轴的一致性较低。这个结果是测量诊断，应先解释模型和语料差异，不按是否得到期望负相关选择模型。原始余弦数值不可直接跨模型作差，图采用共同样本排名比较。

## 4. 锚词、质心与排除检验

中文锚词在查看得分前固定于`semantic_specs/axes_zh_v1.json`。女性—男性为正方向，尊崇—轻视、地位高—低、力量强—弱等分别建立轴。全部锚词在两模型均有显式向量。首轮轴是研究者提出的候选量表，不是已经完成调查验证的中文声望量表。特别是尊重／蔑视等词可能涉及互动待遇和道德评价，高贵／卑贱等词可能涉及身份等级与财富；必须结合独立评价检验构念。

已完成的稳健性版本：删除已标记人物、组织后缀规则词、明确品牌／平台词、带男／女字词、轴锚词，以及合并三组明确别名。人物标记来自原词表的713条已标注人物，剩余词未完全实体分型，因此“删除已标记人物”不能写成“删除所有人物”。组织和品牌排除同样是透明规则子集。别名检验仅合并人工智能/AI、世界卫生组织/世卫、中国乒乓球队/国乒三组，不声称完成全词表实体链接。

每个主轴另做逐组删除锚词检验；性别轴追加亲属词和代词版本。没有把词轴强行正交化，也没有按结果筛选锚词。领域质心以单位化词向量等权平均；报告平均投影与再单位化质心投影两种结果，明确区分含义。多领域词在各领域描述中都出现，但总体相关每词只计一次；另计算控制多标签领域指标后的残差相关。五个领域均值只作描述，不以5个点推断稳定总体关系。

性别锚词替换会显著改变尊崇关联，不能仅凭主轴结果下结论：

{md(['模型','替代性别轴','与尊崇候选轴的相关'],sensitivity_rows)}

这些版本捕捉不同的性别线索，均不能自动充当金标准。

## 5. 对词项测量的实质检查

附表extremes列出各轴两端各15词，供核对语义。例如社会地位轴高端出现奢侈品牌和高尔夫等，提示其可能同时捕捉阶层／财富；尊崇轴高端有公祭、国葬等仪式性词项，说明“尊崇关联”不等价于文化消费的高声望。性别轴部分人名排序与本人性别直觉不符，需要检查同名、多义和语料语境，不能直接把名字投影解释成受众性别。

因此目前最可靠的产物是可审计的词项地图、模型分歧和候选维度诊断。主张文化价值贬抑之前，需要独立的人类评价检验，而不是调锚词直到出现预期关系。

## 6. 行为联结的状态

本地只有五领域汇总包，没有服务器月度逐帖命中Parquet。词项级转发／表达差异不能由五领域系数反推。因此图中的点现在按领域区分，尚未以词项行为性别差着色。需要在服务器基于已保存的retweet_terms、expression_terms重新聚合；现有词表内的词通常无需重新扫描原始文本。不会将领域系数复制给所有词，也不会把热搜标题次数当作微博参与频率。

## 7. 文件

- [六页PDF图集]({a.date}_semantic_figures.pdf)：覆盖、文化地图、领域分布、排除稳健性、共同样本模型一致性、锚词敏感性。
- [{a.date}_semantic_tables]({a.date}_semantic_tables/)：全部词项得分、缺失、质心、相关、极端词与来源哈希。
- `term_semantics.py`：流式提取和测量；`semantic_report.py`：复核、模型比较和图表。
- 数值验证：余弦范围与有限性、单位向量、向量文件声明行数、独立逐元素投影与矩阵计算一致。模型训练不确定性没有因词项自助抽样而自动消失，故本轮不以词项置信区间宣称总体显著性。
'''
    (out/f'{a.date}_semantic_analysis.md').write_text(text,encoding='utf-8')
    print(out/f'{a.date}_semantic_analysis.md')


if __name__=='__main__':main()
