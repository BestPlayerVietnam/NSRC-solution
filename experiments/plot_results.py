"""Export a held-out cost prediction figure from the completed experiment."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import analyze as a
import predictor as pred

def main():
    evaluation=json.loads((a.RUN/'evaluation.json').read_text(encoding='utf-8'))
    primary=set(evaluation['primary_panel']['ids']['test'])
    rows=[r for r in a.all_rows() if r['split']=='test' and r['id'] in primary and a.eligible(r)]
    predictions={(r['id'],r['model']):r['prediction'] for r in a.read_jsonl(a.RUN/'test_predictions.jsonl')}
    colors=['#277DA8','#35A176','#E18B32','#945DB3'];labels=['3.6 plain','3.8 plain','3.6 reasoning','3.8 reasoning']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    fig.suptitle('NSRC | Pre-run token prediction on held-out tasks',fontweight='bold',fontsize=15)
    maes=[];old=[]
    for i,model in enumerate(pred.MODELS):
        rr=[r for r in rows if r['model']==model]
        actual=[r['provider_tokens'] for r in rr];estimated=[predictions[(r['id'],model)]['expected_tokens'] for r in rr]
        axes[0].scatter(actual,estimated,s=30,alpha=.8,c=colors[i],label=f'{labels[i]} (n={len(rr)})',edgecolors='white',linewidths=.4)
        metric=evaluation['models'][model]['splits']['test']['prediction']
        maes.append(metric['cost'].get('mae',float('nan')));old.append(metric['heuristic_cost'].get('mae',float('nan')))
    axes[0].plot([1,5000],[1,5000],color='#737D87',ls='--',lw=1,label='Perfect prediction')
    axes[0].set(xscale='log',yscale='log',xlabel='Observed weighted tokens',ylabel='Predicted weighted tokens',xlim=(5,5000),ylim=(.8,5000))
    axes[0].legend(fontsize=8,loc='upper left',frameon=False);axes[0].grid(True,alpha=.15,which='both')
    x=np.arange(4);width=.36
    axes[1].bar(x-width/2,old,width,color='#CED5DD',label='Initial heuristic')
    bars=axes[1].bar(x+width/2,maes,width,color=colors,label='Frozen predictor')
    for b,v in zip(bars,maes):
        if np.isfinite(v):axes[1].annotate(f'{v:.0f}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=9,xytext=(0,3),textcoords='offset points')
    axes[1].set_xticks(x,labels,rotation=12);axes[1].set_ylabel('Mean absolute error (weighted tokens)');axes[1].legend(frameon=False,fontsize=9);axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.supxlabel('Fixed 20-task primary panel; cost of a bounded solver, not guaranteed success. Smaller MAE is better.',fontsize=9,color='#536172')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.RUN/'prediction_test.png',dpi=170,facecolor='white')
    fig.savefig(a.RUN/'prediction_test.svg',facecolor='white');plt.close(fig)
    print(a.RUN/'prediction_test.png')
if __name__=='__main__':main()
