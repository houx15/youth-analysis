"""核验两版服务器汇总包并输出阶段报告、日期前缀PDF与小型审计表。

只读取汇总表；分布文件分块过滤，不加载原始微博或用户明细。
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from cultural_participation.result_report import DOMAINS, LABELS, ZH, COLORS, MARKERS

VERSIONS=['exclusive_full_v1','exclusive_top250_v1']
NAMES=['Exclusive full','Top 250 per domain']
KEY=['period','sample','stratum','level','gender','domain','metric']


def load(path):
    data={};audit={}
    with tarfile.open(path) as tar:
        sample={name:pd.read_csv(tar.extractfile(f'sample/{name}.csv')) for name in ['matching_balance','profile_coverage','sample_coverage','stratified_balance','age_distribution']}
        sample_meta=json.load(tar.extractfile('sample/manifest.json'))
        if sample_meta['status']!='complete':raise ValueError('匹配未完成')
        for version in VERSIONS:
            root=f'{version}/extended_summary/'
            d=pd.read_csv(tar.extractfile(root+'descriptives.csv'))
            gaps=pd.read_csv(tar.extractfile(root+'gender_gaps.csv'))
            meta=json.load(tar.extractfile(root+'manifest.json'))
            annual=pd.read_csv(tar.extractfile(f'{version}/annual_summary/gender_gaps.csv'))
            annual_meta=json.load(tar.extractfile(f'{version}/annual_summary/manifest.json'))
            if meta['status']!='complete' or meta['sample_manifest']!=sample_meta:raise ValueError('来源或样本不一致')
            sources=meta['source']
            if len(sources)!=12 or {m['signature']['month'] for m in sources}!=set(range(1,13)):raise ValueError('月份清单不完整')
            if any(m['status']!='complete' for m in sources):raise ValueError('月份未完成')
            hashes={m['signature']['vocabulary_sha256'] for m in sources}
            current=hashlib.sha256((Path(__file__).parent/'vocabularies'/version/'vocabulary.json').read_bytes()).hexdigest()
            if hashes!={current} or annual_meta['source_manifests']!=sources:raise ValueError('词表或年度结果版本不同')
            if d.duplicated(KEY).any():raise ValueError('描述键重复')
            means=d.pivot(index=[c for c in KEY if c!='gender'],columns='gender',values='mean')
            expected=(means.f-means.m).rename('female_minus_male')
            actual=gaps.set_index([c for c in KEY if c!='gender']).female_minus_male
            pd.testing.assert_series_equal(expected.sort_index(),actual.sort_index(),check_names=False,atol=1e-9,rtol=1e-9)
            count=d[(d.stratum=='all')&d.metric.isin(['expressions','retweets'])]
            for (s,g,dom,metric),sub in count.groupby(['sample','gender','domain','metric']):
                months=sub[sub.period!='annual_user'];year=sub[sub.period=='annual_user'].iloc[0]
                if len(months)!=12 or months.n_users.nunique()!=1 or months.n_users.iloc[0]!=year.n_users:raise ValueError('零月份或分母不一致')
                np.testing.assert_allclose(months['mean'].mean(),year['mean'],atol=1e-9)
            # 逐块核验直方图频数守恒；仅保留年度用户分布供本地作图。
            distribution=[];summed=[]
            with tar.extractfile(root+'distributions.csv') as stream:
                for chunk in pd.read_csv(stream,chunksize=50000):
                    summed.append(chunk.groupby(KEY)['n'].sum())
                    distribution.append(chunk[chunk.period=='annual_user'])
            totals=pd.concat(summed).groupby(level=list(range(len(KEY)))).sum().sort_index()
            pd.testing.assert_series_equal(totals,d.set_index(KEY).n_users.sort_index(),check_names=False,check_dtype=False)
            data[version]={'d':d,'gaps':gaps,'dist':pd.concat(distribution),'annual':annual}
            audit[version+'_month_files']={m['signature']['month']:len(m['signature']['inputs']) for m in sources}
            audit[version]={'vocabulary_sha256':current,'source_git':sources[0]['signature']['git_sha'],'unique_posts':annual_meta['unique_posts'],'sample_sha256':meta['sample_sha256']}
    if audit[VERSIONS[0]]['sample_sha256']!=audit[VERSIONS[1]]['sample_sha256']:raise ValueError('两版本匹配用户不同')
    audit['input_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    audit['checks']=['vocabulary hashes','12 complete month manifests','same matching sample','female-minus-male identity','monthly frequency annual identity','histogram count conservation']
    return data,sample,audit


def style(ax):
    ax.grid(alpha=.2,linewidth=.5);ax.set_axisbelow(True)


def finish(fig,title,note):
    fig.suptitle(title,fontsize=13);fig.text(.5,.015,note,ha='center',fontsize=8)
    fig.text(.5,.925,'HISTORICAL SAMPLE | Source population unverified; no population-wide inference',
             ha='center',fontsize=9,color='#9b3333')
    fig.tight_layout(rect=(0,.045,1,.90));return fig


def annual(data,version,sample='matched',stratum='all'):
    d=data[version]['d'];return d[(d.period=='annual_user')&(d['sample']==sample)&(d.stratum==stratum)]


def figures(data,sample):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    b=sample['matching_balance'];b=b[b.scale=='log1p']
    for name,marker,color in [('all','o','#999999'),('matched','s','#222222')]:
        sub=b[b['sample']==name];axes[0].plot(sub.smd,range(3),marker=marker,color=color,label=name)
    axes[0].set_yticks(range(3),['Total posts','Total retweets','Active days']);axes[0].axvline(0,color='black',lw=.5);axes[0].set_xlabel('Standardized mean difference (female - male)');axes[0].legend();style(axes[0])
    p=sample['profile_coverage'].set_index('gender')
    for i,sex in enumerate(['m','f']):
        axes[1].bar(i,p.loc[sex,'n_users'],color=COLORS[sex],alpha=.25)
        axes[1].bar(i,p.loc[sex,'n_matched'],color=COLORS[sex]);axes[1].text(i,p.loc[sex,'n_users'],f"{int(p.loc[sex,'n_matched']):,} / {int(p.loc[sex,'n_users']):,}",ha='center',va='bottom')
    axes[1].set_xticks([0,1],['Male','Female']);axes[1].set_ylabel('Users: matched / original');axes[1].set_ylim(0,p.n_users.max()*1.16);style(axes[1])
    yield '01_matching',finish(fig,'Activity matching | Same matched sample for both vocabularies','1:1 coarsened exact matching on three log1p activity measures. Shaded bars: original sample.')
    fig,axes=plt.subplots(2,5,figsize=(14,6))
    combinations=[(VERSIONS[0],'all'),(VERSIONS[0],'matched'),(VERSIONS[1],'matched')]
    for row,metric in enumerate(['expressions','retweets']):
        for col,domain in enumerate(DOMAINS):
            ax=axes[row,col]
            for sex in ['m','f']:
                y=[annual(data,v,s).query('domain == @domain and metric == @metric and gender == @sex',local_dict={'domain':domain,'metric':metric,'sex':sex})['mean'].iloc[0] for v,s in combinations]
                ax.plot(range(3),y,color=COLORS[sex],marker=MARKERS[sex],label=sex)
            ax.set_xticks(range(3),['Full\nall','Full\nmatched','250\nmatched']);ax.set_title(LABELS[col],fontsize=9);ax.set_ylim(bottom=0);style(ax)
            if col==0:ax.set_ylabel(f'{metric.capitalize()} per user-month');ax.legend()
    yield '02_frequency',finish(fig,'Monthly frequency | Activity matching and equal-size vocabulary','Observed annual count / 12 for each user; zero counts retained. Points are means, not confidence intervals.')
    fig,axes=plt.subplots(2,5,figsize=(14,6))
    for row,metric in enumerate(['expressions','retweets']):
        for col,domain in enumerate(DOMAINS):
            ax=axes[row,col]
            for v,linestyle in zip(VERSIONS,['-','--']):
                d=data[v]['d'];sub=d[(d.stratum=='all')&(d['sample']=='matched')&(d.period!='annual_user')&(d.domain==domain)&(d.metric==metric)].copy();sub['month']=sub.period.str[-2:].astype(int)
                for sex in ['m','f']:
                    g=sub[sub.gender==sex].sort_values('month').copy();g.loc[g.month==10,'mean']=np.nan;ax.plot(g.month,g['mean'],color=COLORS[sex],ls=linestyle,marker=MARKERS[sex],ms=3)
            ax.axvspan(9.5,10.5,color='gray',alpha=.15);ax.set_title(LABELS[col],fontsize=9);ax.set_xticks([1,4,7,10,12]);ax.set_ylim(bottom=0);style(ax)
            if col==0:ax.set_ylabel(metric.capitalize()+' / user-month')
    yield '03_monthly',finish(fig,'Monthly trajectories | Matched users','Blue circles: male; orange triangles: female. Solid: full; dashed: 250. October has no source files (gap); June, September and November have partial coverage.')
    fig,axes=plt.subplots(2,5,figsize=(14,6))
    for row,metric in enumerate(['expressions','retweets']):
        for col,domain in enumerate(DOMAINS):
            ax=axes[row,col]
            for v,ls in zip(VERSIONS,['-','--']):
                h=data[v]['dist'];h=h[(h.stratum=='all')&(h['sample']=='matched')&(h.metric==metric)&(h.domain==domain)&np.isfinite(h.upper)&(h.upper<=100)]
                for sex in ['m','f']:
                    g=h[h.gender==sex].sort_values('upper');ax.plot(g.upper,g.cumulative,color=COLORS[sex],ls=ls)
            ax.set_xscale('symlog',linthresh=1);ax.set_xlim(0,100);ax.set_ylim(0,1.02);ax.set_title(LABELS[col],fontsize=9);style(ax)
            if col==0:ax.set_ylabel(metric.capitalize()+' cumulative share')
    yield '04_distributions',finish(fig,'Frequency distributions | Annual mean monthly frequency per user','CDF at histogram upper bounds (strictly below bound). Solid: full; dashed: 250. Tail above 100 not shown.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric in zip(axes,['retweet_share','expression_share']):
        for i,v in enumerate(VERSIONS):
            a=data[v]['annual'];a=a[(a.metric==metric)&(a.model=='activity_adjusted_OLS')].set_index('domain').loc[DOMAINS]
            y=np.arange(5)+(i-.5)*.18
            ax.errorbar(a.estimate_f_minus_m*100,y,xerr=np.vstack([(a.estimate_f_minus_m-a.ci_low)*100,(a.ci_high-a.estimate_f_minus_m)*100]),fmt=['o','s'][i],color=['#999999','#222222'][i],label=NAMES[i],capsize=2)
        ax.set_yticks(range(5),LABELS);ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Female - male (percentage points)');ax.set_title(metric);style(ax)
    axes[0].legend(fontsize=8)
    yield '05_share_gaps',finish(fig,'Participation shares | Full-sample activity-adjusted OLS','95% HC1 intervals from server estimates. These are full-sample regressions, separate from matching.')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,metric in enumerate(['expressions','retweets']):
        matrices=[]
        for v in VERSIONS:
            a=annual(data,v,stratum='region');a=a[a.metric==metric].pivot(index='level',columns=['domain','gender'],values='mean')
            matrices.append(pd.DataFrame({d:a[(d,'f')]-a[(d,'m')] for d in DOMAINS}).reindex(['East','Central','West','Northeast','Unknown']))
        limit=max(np.nanmax(np.abs(m.to_numpy())) for m in matrices)
        for col,(v,m) in enumerate(zip(VERSIONS,matrices)):
            ax=axes[row,col];ax.imshow(m,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(5),['Politics','Business','Technology','Sports','Arts'],rotation=15);ax.set_yticks(range(5),m.index);ax.set_title(f'{NAMES[col]} | {metric}')
            for i in range(5):
                for j in range(5):ax.text(j,i,f'{m.iloc[i,j]:+.2f}',ha='center',va='center',color='white' if abs(m.iloc[i,j])>limit*.6 else 'black',fontsize=9)
    yield '06_regions',finish(fig,'Region differences | Female - male monthly frequency, matched sample','Shared color scale within each metric. Region Unknown is retained; region differences are descriptive.')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,divisor,label in zip(axes,['delay_seconds','log_delay'],[86400,1],['Mean interval (days)','Mean log(1 + seconds)']):
        for i,sex in enumerate(['m','f']):
            for j,v in enumerate(VERSIONS):
                a=annual(data,v);a=a[(a.metric==metric)&(a.gender==sex)].set_index('domain').loc[DOMAINS]
                ax.plot(a['mean']/divisor,np.arange(5)+(i-.5)*.18,marker=MARKERS[sex],color=COLORS[sex],ls=['-','--'][j])
        ax.set_yticks(range(5),LABELS);ax.set_xlabel(label);style(ax)
    yield '07_intervals',finish(fig,'Repost intervals | User-weighted annual means, matched sample','Solid: full; dashed: 250. Arithmetic means and mean log intervals reveal different parts of a long-tailed distribution.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    a=sample['age_distribution']
    for sex in ['m','f']:
        d=a[a.gender==sex];axes[0].plot(d.age,d.n_users/d.n_users.sum()*100,color=COLORS[sex],marker=MARKERS[sex],label=sex)
    axes[0].set_xlabel('2020 - recorded birth year (UNVERIFIED)');axes[0].set_ylabel('Within-gender percent');axes[0].legend();style(axes[0])
    b=sample['stratified_balance'];b=b[(b.stratum=='age_group')&(b['sample']=='matched')]
    for variable,g in b.groupby('variable'):axes[1].plot(g.level,g.smd,marker='o',label=variable)
    axes[1].axhline(.1,color='gray',ls='--');axes[1].axhline(-.1,color='gray',ls='--');axes[1].set_ylabel('Within-group log-activity SMD');axes[1].legend(fontsize=8);style(axes[1])
    yield '08_age_diagnostic',finish(fig,'Age diagnostics | Sample provenance unresolved','All 225,339 users have recorded ages 0-16. Verify upstream selection and birthday semantics before generalizing any result.')


def md(df):
    return '|'+ '|'.join(map(str,df.columns))+'|\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('|'+ '|'.join(map(str,row))+'|' for row in df.itertuples(index=False,name=None))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='cultural_participation/outputs/exclusive_review.tar.gz');p.add_argument('--date',default='20261016');p.add_argument('--output',default='cultural_participation/reports');a=p.parse_args()
    data,sample,audit=load(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    table_dir=out/f'{a.date}_exclusive_tables';table_dir.mkdir(exist_ok=True)
    for name,table in sample.items():table.to_csv(table_dir/f'{name}.csv',index=False)
    for v,tables in data.items():
        monthly=tables['d']
        monthly=monthly[(monthly.period!='annual_user')&(monthly.stratum=='all')].copy()
        month_numbers=monthly.period.str[-2:].astype(int)
        monthly['source_file_count']=month_numbers.map(audit[v+'_month_files'])
        monthly['observation_status']=np.select(
            [month_numbers.eq(10),month_numbers.isin([6,9,11])],
            ['no_observation','partial_coverage'],default='complete_by_manifest')
        monthly.to_csv(table_dir/f'{v}_monthly.csv',index=False)
        for name,t in tables.items():
            if name in ['d','gaps']:t=t[t.period=='annual_user']
            if name=='dist':t=t[t.stratum=='all']
            t.to_csv(table_dir/f'{v}_{name}.csv',index=False)
    with PdfPages(out/f'{a.date}_exclusive_figures.pdf') as pdf:
        for name,fig in figures(data,sample):
            fig.savefig(out/f'{a.date}_exclusive_{name}.pdf',bbox_inches='tight')
            fig.savefig(out/f'{a.date}_exclusive_{name}.png',dpi=130,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    rows=[]
    for domain in DOMAINS:
        row={'领域':ZH[domain]}
        for v,label in zip(VERSIONS,['全量','250词']):
            d=annual(data,v)
            for metric,lab in [('expressions','表达'),('retweets','转发')]:
                x=d[(d.domain==domain)&(d.metric==metric)].set_index('gender')['mean']
                row[f'{label}{lab}男/女']=f'{x.m:.3f} / {x.f:.3f}'
        rows.append(row)
    b=sample['matching_balance'];age=sample['age_distribution'];profile=sample['profile_coverage'].set_index('gender')
    maxsmd=b[b['sample']=='matched'].smd.abs().max();invalid_fraction=age[age.age<10].n_users.sum()/age.n_users.sum()
    matches=int(profile.n_matched.iloc[0]);young_balance=sample['stratified_balance'].query("stratum == 'age_group' and sample == 'matched'").smd.abs().max()
    expected_hash='239c11861627840a6478cb837c74e45e693c69c4825e99aa15dad314f204744a'
    # 本阶段解释对应当前结果包，换数据时必须重新审阅叙述。
    if audit['input_sha256']!=expected_hash:raise ValueError('结果包变更，请复核本阶段文字解释后更新输入哈希')
    audit['interpretation_status']='historical_sample_population_unverified'
    audit['scope_note']='全部225339用户按外接生日为0–16岁；样本来源待核实，全部结果不得外推为全量用户结论。'
    report=f'''# 互斥词表两版本：历史样本行为分析阶段报告

> 归档状态：样本范围待核实。全部225,339用户按外接生日计算均为0–16岁。这关系到整批用户的来源与选择机制，影响所有总体、匹配、地区及时间结果的外推资格，不能仅作为年龄字段问题处理。本文“全量词表”仅指3,838词，“全部样本”仅指本包用户，均不表示全量微博用户。

日期：{a.date}。数据为2020年观测微博。全量互斥词表3,838词；稳健性词表各域250词，共1,250词。词频为独立热搜标题数。两版均从原始微博重新匹配。

清单确认实际输入为`/gpfs/share/home/2401111059/youth_analysis/cleaned_weibo_cov/2020`。本地旧上游代码指向youth筛选链，但尚不能证明服务器实际文件如何生成。用户说明后来已改为全量；这项说明尚未与本包输入建立可核验的生成链。另行计划从`/gpfs/share/home/2401111059/cleaned_weibo_data`重建2020—2025年数据，先做2020；本归档没有执行该重建。

## 已核验的数据与匹配

两版输入清单均为12个月、各303个输入文件，年度去重144,934,465帖；分析225,339用户。缺少的观测日期沿用既定数据范围，没有补造帖子。两版使用相同匹配名单。

活动量匹配保留{matches:,}对用户（共{matches*2:,}人），男性保留率{matches/profile.loc['m','n_users']:.2%}，女性{matches/profile.loc['f','n_users']:.2%}。总发帖、总转发、活跃天数的原始／log尺度匹配后SMD最大绝对值{maxsmd:.4f}；这三项均值平衡良好。匹配样本的总体活动水平与原始女性总体不同，结论对应保留的共同活动范围。

## 月度频次：核心方向在等规模词表与匹配后保留

下表为匹配后每用户全年次数除12，先在用户内计算，再分别按男女平均。未命中记零；全年计数是已有观测日期上的计数，除12作统一尺度。

{md(pd.DataFrame(rows))}

两个版本中，男性在政治、商业、科技、体育的表达与转发频次均更高，女性在文艺娱乐更高。等规模词表下方向保持，但绝对次数及差距有所改变。因此可说总体性别分化方向对本次等规模高频词检验稳健，不能说词表规模完全不影响数值。科技251词缩至250词，几乎未改变覆盖；其他领域缩减更大。

## 分布与月份

图2给出原始样本、匹配全量词表、匹配250词三个版本的频次。图3给出12个月变化，图4给出年度用户月均次数的分箱累计分布：均值不能替代完整分布。分布图保留零值，展示至月均100次，长尾数据仍保存在CSV。图中的CDF来自预设直方图区间，不是精确分位数；不报告从图上猜测的精确中位数。沿用用户确认的日期缺口：6月29个文件、9月5个、10月0个、11月24个；其他月份完整。10月在曲线中断开并标灰，不把无观测画成零活动；部分覆盖月份不用于解释季节性低谷。本轮不因日期缺口停止其他比较。

## 参与份额与时间间隔

全体样本的活动量调整OLS也保持前四领域男性份额更高、文艺女性份额更高的模式，图5展示两版系数及HC1区间。该回归与匹配是不同的比较方案。

时间间隔需要同时看算术平均秒数和用户平均log(1+秒)。匹配全量版本中，文艺的平均log间隔几乎相同；算术均值却略为女性更长。商业则女性算术平均间隔更短、平均log间隔更长。政治的算术均值方向还随词表版本改变。因此本轮不将所有时间结果概括为统一的“女性在前四领域更慢、文艺更快”；这些结果反映间隔分布及长尾的差异。图7并列展示两种统计量。

## 地区分层

东部、中部、西部、东北四个已知大区，在两版匹配样本中都保留了前四领域男性频次更高、文艺女性更高的方向。差距大小随地区变化；例如全量版本的政治转发月频次女性减男性，东部约-2.04，东北约-5.58。这里只报告分层描述，不据此给出地区间差距显著性排名。

男性地区可用率{profile.loc['m','n_region_known']/profile.loc['m','n_users']:.2%}，女性{profile.loc['f','n_region_known']/profile.loc['f','n_users']:.2%}；Unknown组单列。已知四大区的匹配后log活动量SMD绝对值均低于0.10，但Unknown组总转发约-0.121，仍有残余差异。

## 年龄诊断揭示样本来源问题

按外接birthday取2020减出生年后，全部225,339用户的记录年龄为0–16岁，其中{invalid_fraction:.2%}为0–9岁，85.19%为0–13岁。需要同时核实样本筛选链、生日字段含义及连接过程；目前既不能确认这些是真实年龄，也不能确认样本代表全量用户或真实未成年人总体。本报告保留单岁分布和分组表供来源诊断，暂不解释年龄组的社会学差异。

此外，全局匹配不保证年龄组内匹配：14–15、16–17记录组的log转发量SMD仍约0.282、0.315；年龄组内最大活动量SMD为{young_balance:.3f}。即使生日含义得到确认，也不能将目前这些组称为活动量已充分平衡的年龄组。

## 当前结论与后续边界

本轮仅支持对这批来源待核实用户的描述：在互斥词表、等规模高频词表及总体活动量匹配下，男女的领域表达／转发配置方向保持一致；已知地区分层也重复这一方向。强度并非完全不变，时间间隔也没有同样整齐的模式。词表检验与活动量平衡不能修复样本选择偏差。数值计算可留作历史记录，但全部结果的总体适用范围均未确立，不得表述为2020年全量微博用户、中国青年或真实未成年人总体的结论。

语义、声望与性别观念的测量不在本包内；本报告不将行为差异转换为价值贬抑结论。匹配／地区频次图是描述性均值，不含未经配对估计的差异区间。

## 产物

- [中文阶段归档总结]({a.date}_exclusive_stage_summary.md)
- [八页PDF图集]({a.date}_exclusive_figures.pdf)
- [小型结果与诊断表]({a.date}_exclusive_tables/)
- [输入与一致性核验]({a.date}_exclusive_audit.json)

完整月度表和直方图保留于原始汇总压缩包。代码：`python -m cultural_participation.exclusive_report`。输入SHA256：`{audit['input_sha256']}`。
'''
    (out/f'{a.date}_exclusive_analysis.md').write_text(report,encoding='utf-8')
    (out/f'{a.date}_exclusive_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(out/f'{a.date}_exclusive_analysis.md')


if __name__=='__main__':main()
