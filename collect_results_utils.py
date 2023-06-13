import json
from glob import glob
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def collect_results(task, models, languages, metric, seeds=[2001, 2002, 2003]):
    all_results = []
    for model, seed, lang_src, lang_tgt in itertools.product(models, seeds, languages, languages):
        path = f'models/{task}/{model}_seed{seed}/{lang_src}/{lang_tgt}/eval_results.json'
        if not os.path.exists(path):
            print(f'File not found: {path}')
            continue
        with open(path) as f:
            results = json.load(f)
        all_results.append({
            'model': model,
            'seed': seed,
            'lang_src': lang_src,
            'lang_tgt': lang_tgt,
            metric: results[f'eval_{metric}'],
        })
    return all_results


# Bootstrap using more combinations of seeds
def show_inlanguage(results, task, metric):
    df = pd.DataFrame(results)
    
    df = df[df['lang_src'] == df['lang_tgt']]
    summary = df.set_index(["model", "lang_src", "lang_tgt"]).groupby(["model"]).mean()

    # generate 100 bootstrapping samples
    bootstrapped_means = []
    options = {}
    for model in df['model'].unique():
        for lang in df['lang_tgt'].unique():
            options[(model, lang)] = df[(df['model'] == model) & (df['lang_tgt'] == lang)]
    for model in df['model'].unique():
        for sample_i in range(100):
            sum_metric = 0
            for lang in df['lang_tgt'].unique():
                sample = options[(model, lang)].sample(n=1, replace=True)
                sum_metric += sample[metric].values[0]
            bootstrapped_means.append({
                'model': model,
                'seed': sample_i,
                metric: sum_metric / len(df['lang_tgt'].unique())
            })

    df_bootstrap = pd.DataFrame(bootstrapped_means)
    # display(df_bootstrap)
    # df = 

    min_val = summary[metric].min()
    max_val = summary[metric].max()
    delta = (max_val - min_val)
    min_val = min_val - delta
    max_val = max_val + delta
    # display(df.sort_values(metric))
    sns.barplot(x='model', y=metric, data=df_bootstrap, errorbar="sd")
    plt.ylim(min_val, max_val)
    plt.title(f'{task} In-language {metric} (all languages)')
    plt.xticks(rotation=75)
    plt.show()

def show_inlanguage_per_lang(results, task, metric, tok_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] == df['lang_tgt']]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_result', rsuffix='_tok')
    # sort by data rank
    df = df.sort_values(["data_amount_rank", "model"])
    g = sns.lineplot(data=df, x="lang_tgt", y=metric, hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} by language and model")
    plt.show()

    df_baseline = df[df["model"] == "multilingual_unigram_alpha0.3"].drop(columns=["model", "lang_src"]).groupby(["lang_tgt"]).mean().reset_index()
    df_delta = df.join(df_baseline.set_index("lang_tgt"), on="lang_tgt", rsuffix="_baseline")
    df_delta[metric+"_delta"] = df_delta[metric] - df_delta[metric+"_baseline"]
    # plot the delta
    sns.lineplot(data=df_delta, x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} delta over the baseline by language and model")
    plt.show()

    if len(df["model"].unique()) > 6:
        # plot the delta
        for model_set in ["Chung", "beta", "k_"]:
            sns.lineplot(data=df_delta[df_delta['model'].str.contains(model_set) | df_delta['model'].str.contains('unigram')], x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
            plt.title(f"{task} {metric} delta over the baseline by language and model")
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.show()


import scipy.stats as stats

def show_scatter(results, x, y, tok_df, tok_cross_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] == df["lang_tgt"]]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')
    df = df.join(tok_cross_df.set_index(['model', 'lang_src', 'lang_tgt']), on=['model', 'lang_src', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')

    mean = df.drop(columns=["model", "lang_tgt"]).groupby(["lang_src"]).mean()
    delta = df.join(mean, on="lang_src", rsuffix="_mean")
    delta = delta.sort_values(["lang_src", "model"])
    delta[y+"_delta"] = delta[y] - delta[y + "_mean"]
    delta[x+"_delta"] = delta[x] - delta[x + "_mean"]
    sns.scatterplot(data=delta, x=x+"_delta", y=y+"_delta", hue="model", style="model", markers=True)
    plt.title(f"{y} vs {x} adjusted by language mean")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    print("pearson", stats.pearsonr(x=delta[x+"_delta"], y=delta[y+"_delta"]))
    print("spearman", stats.spearmanr(a=delta[x+"_delta"], b=delta[y+"_delta"]))
    
# Bootstrap using more combinations of seeds
def show_crosslanguage(results, task, metric):
    df = pd.DataFrame(results)
    
    df = df[df['lang_src'] == "en"]
    summary = df.set_index(["model", "lang_src", "lang_tgt"]).groupby(["model"]).mean()

    # generate 100 bootstrapping samples
    bootstrapped_means = []
    options = {}
    for model in df['model'].unique():
        for lang in df['lang_tgt'].unique():
            options[(model, lang)] = df[(df['model'] == model) & (df['lang_tgt'] == lang)]
    for model in df['model'].unique():
        for sample_i in range(300):
            sum_metric = 0
            for lang in df['lang_tgt'].unique():
                sample = options[(model, lang)].sample(n=1, replace=True)
                sum_metric += sample[metric].values[0]
            bootstrapped_means.append({
                'model': model,
                'seed': sample_i,
                metric: sum_metric / len(df['lang_tgt'].unique())
            })

    df_bootstrap = pd.DataFrame(bootstrapped_means)

    min_val = summary[metric].min()
    max_val = summary[metric].max()
    delta = (max_val - min_val)
    min_val = min_val - delta
    max_val = max_val + delta
    # display(df.sort_values(metric))
    sns.barplot(x='model', y=metric, data=df_bootstrap, errorbar="sd")
    plt.ylim(min_val, max_val)
    plt.title(f'{task} Cross-language {metric} (English -> *)')
    plt.xticks(rotation=75)
    plt.show()

def show_crosslanguage_per_lang(results, task, metric, tok_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] == "en"]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_result', rsuffix='_tok')
    # sort by data rank
    df = df.sort_values(["data_amount_rank", "model"])
    g = sns.lineplot(data=df, x="lang_tgt", y=metric, hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} - transfer from english by target language and model")
    plt.show()

    df_baseline = df[df["model"] == "multilingual_unigram_alpha0.3"].drop(columns=["model", "lang_src"]).groupby(["lang_tgt"]).mean().reset_index()
    df_delta = df.join(df_baseline.set_index("lang_tgt"), on="lang_tgt", rsuffix="_baseline")
    df_delta[metric+"_delta"] = df_delta[metric] - df_delta[metric+"_baseline"]
    # plot the delta
    sns.lineplot(data=df_delta, x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} - transfer from english - delta over the baseline by target language and model")
    plt.show()

    if len(df["model"].unique()) > 6:
        for model_set in ["Chung", "beta", "k_"]:
            sns.lineplot(data=df_delta[df_delta['model'].str.contains(model_set) | df_delta['model'].str.contains('unigram')], x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
            plt.title(f"{task} {metric} - transfer from english delta over the baseline by language and model")
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.show()

def show_scatter_crossling(results, x, y, tok_df, tok_cross_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] == "en"]
    df = df[df["lang_tgt"] != "en"]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')
    df = df.join(tok_cross_df.set_index(['model', 'lang_src', 'lang_tgt']), on=['model', 'lang_src', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')

    mean = df.drop(columns=["model", "lang_src"]).groupby(["lang_tgt"]).mean()
    delta = df.join(mean, on="lang_tgt", rsuffix="_mean")
    delta[y+"_delta"] = delta[y] - delta[y + "_mean"]
    delta[x+"_delta"] = delta[x] - delta[x + "_mean"]
    sns.scatterplot(data=delta, x=x+"_delta", y=y+"_delta", hue="model", style="model", markers=True)
    plt.title(f"{y} vs {x} adjusted by language mean")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    print("pearson", stats.pearsonr(x=delta[x+"_delta"], y=delta[y+"_delta"]))
    print("spearman", stats.spearmanr(a=delta[x+"_delta"], b=delta[y+"_delta"]))


# Bootstrap using more combinations of seeds
def show_crosslanguage_all(results, task, metric):
    df = pd.DataFrame(results)
    
    df = df[df['lang_src'] != df['lang_tgt']]
    summary = df.set_index(["model", "lang_src", "lang_tgt"]).groupby(["model"]).mean()

    # generate 100 bootstrapping samples
    bootstrapped_means = []
    options = {}
    for model in df['model'].unique():
        for lang_src in df['lang_src'].unique():
            for lang_tgt in df['lang_tgt'].unique():
                options[(model, lang_src, lang_tgt)] = df[(df['model'] == model) & (df['lang_src'] == lang_src) & (df['lang_tgt'] == lang_tgt)]
    for model in df['model'].unique():
        for sample_i in range(30):
            sum_metric = 0
            total = 0
            for lang_src in df['lang_src'].unique():
                for lang_tgt in df['lang_tgt'].unique():
                    if lang_src == lang_tgt:
                        continue
                    sample = options[(model, lang_src, lang_tgt)].sample(n=1, replace=True)
                    sum_metric += sample[metric].values[0]
                    total += 1
            bootstrapped_means.append({
                'model': model,
                'seed': sample_i,
                metric: sum_metric / total
            })

    df_bootstrap = pd.DataFrame(bootstrapped_means)

    min_val = summary[metric].min()
    max_val = summary[metric].max()
    delta = (max_val - min_val)
    min_val = min_val - delta
    max_val = max_val + delta
    # display(df.sort_values(metric))
    sns.barplot(x='model', y=metric, data=df_bootstrap, errorbar="sd")
    plt.ylim(min_val, max_val)
    plt.title(f'{task} Cross-language {metric} (* -> *)')
    plt.xticks(rotation=75)
    plt.show()

def show_crosslanguage_all_per_lang(results, task, metric, tok_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] != df['lang_tgt']]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_result', rsuffix='_tok')
    # sort by data rank
    df = df.sort_values(["data_amount_rank", "model"])
    g = sns.lineplot(data=df, x="lang_tgt", y=metric, hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} - transfer *->* by target language and model")
    plt.show()

    df_baseline = df[df["model"] == "multilingual_unigram_alpha0.3"].drop(columns=["model"]).groupby(["lang_src", "lang_tgt"]).mean().reset_index()
    df_delta = df.join(df_baseline.set_index(["lang_src", "lang_tgt"]), on=["lang_src", "lang_tgt"], rsuffix="_baseline")
    df_delta[metric+"_delta"] = df_delta[metric] - df_delta[metric+"_baseline"]
    # plot the delta
    sns.lineplot(data=df_delta, x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{task} {metric} - transfer *->* delta over the baseline by target language and model")
    plt.show()

    if len(df["model"].unique()) > 6:
        for model_set in ["Chung", "beta", "k_"]:
            sns.lineplot(data=df_delta[df_delta['model'].str.contains(model_set) | df_delta['model'].str.contains('unigram')], x="lang_tgt", y=metric+"_delta", hue="model", style="model", markers=True, dashes=False)
            plt.title(f"{task} {metric} - transfer *->* delta over the baseline by language and model")
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.show()

def show_scatter_crossling_all(results, x, y, tok_df, tok_cross_df):
    df = pd.DataFrame(results)
    df = df[df['lang_src'] != df["lang_tgt"]]
    # join with tok_df
    df = df.join(tok_df.set_index(['model', 'lang']), on=['model', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')
    df = df.join(tok_cross_df.set_index(['model', 'lang_src', 'lang_tgt']), on=['model', 'lang_src', 'lang_tgt'], how='left', lsuffix='_results', rsuffix='_tok')

    mean = df.drop(columns=["model"]).groupby(["lang_src", "lang_tgt"]).mean()
    delta = df.join(mean, on=["lang_src", "lang_tgt"], rsuffix="_mean")
    delta[y+"_delta"] = delta[y] - delta[y + "_mean"]
    delta[x+"_delta"] = delta[x] - delta[x + "_mean"]
    sns.scatterplot(data=delta, x=x+"_delta", y=y+"_delta", hue="model", style="model", markers=True)
    plt.title(f"{y} vs {x} adjusted by language mean")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    r = stats.pearsonr(x=delta[x+"_delta"], y=delta[y+"_delta"])[0]
    print("pearson", stats.pearsonr(x=delta[x+"_delta"], y=delta[y+"_delta"]))
    print("spearman", stats.spearmanr(a=delta[x+"_delta"], b=delta[y+"_delta"]))
