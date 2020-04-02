# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random

from gensim.models import KeyedVectors
import pkuseg
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from pyltp import SentenceSplitter
import logging
import re
from sklearn.preprocessing import  MinMaxScaler
from IPython.display import display, HTML
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

seg = pkuseg.pkuseg(postag=True)
with open('data/stopwords.txt', 'r',encoding='utf8') as f:
    stopwords = set([w.strip() for w in f])
small_vec_model = KeyedVectors.load_word2vec_format('data/small_ailab_embedding.txt')

def split_document(document):
    return [sen for sen in SentenceSplitter.split(document) if sen]


def query(words):
    # exist_words = list(filter(lambda x: x in self.vector_from_text, words))
    # non_exist_words = [i for i in words if i not in self.vector_from_text]
    # if non_exist_words:
    #     chars = list(filter(lambda x: x in self.vector_from_text, ''.join(non_exist_words)))
    #     exist_words += chars
    words = [w for w in words if w in small_vec_model]
    words_vector = np.mean([small_vec_model[w] for w in words], axis=0) \
        if words else np.zeros(small_vec_model.vector_size)
    return words_vector


def split_func(string):
    useful_words = [word for word,flag in seg.cut(string) if flag.startswith('n') or flag.startswith('v')]
    return useful_words


def get_sentnce_vector(all_sentences_words):
    sentence_vec = np.array([query(words) for words in all_sentences_words])
    return sentence_vec


def calc_page_rank(sentence_vec):
    sim_mat = cosine_similarity(sentence_vec)
    np.fill_diagonal(sim_mat, 0)
    nx_graph = nx.from_numpy_array(sim_mat)
    tol, max_iter = 1e-7, 1000
    Flag = True
    while Flag:
        try:
            pagerank_score = nx.pagerank(nx_graph, tol=tol, max_iter=max_iter)
            Flag = False
        except nx.PowerIterationFailedConvergence as e:
            print(e)
            tol *= 10
    pagerank_score = np.array([v for k, v in sorted(pagerank_score.items(), key=lambda x: x[0])])
    return pagerank_score


def get_title_similarity(sentence_vec, title_vec):
#     title_vector = get_sentnce_vector(title)
    sim_mat = cosine_similarity(sentence_vec,title_vec)
    return sim_mat

def get_title_common_score(all_sentences_words, title_words):
    set_title_words = set(title_words)    
    ret = []
    for words in all_sentences_words:
        set_words = set(words)& set_title_words
        if len(set_words)>=2:
            ret.append(1.5)
        else:
            ret.append(1)
    return np.array(ret)


def get_position_score(sen_length):
    position_score = np.ones(sen_length)
    position_score[1] = 2 
    position_score[-1] = 1.3
    return position_score

def have_date(sentence):
    if re.findall('[0-9去上前明后]{1,4}年', sentence):
        return True
    if re.findall('[0-9上个一二三四五六七八九十]{1,2}月', sentence):
        return True
    if re.findall('[0-9上昨前]{1,4}日', sentence):
        return True
    if re.findall('[昨|前]天', sentence):
        return True
    return False


with open('data/important_people_orgnazation.txt', 'r', encoding='utf8') as f:
    people_org_set = set()
    for line in f:
        words = line.strip().split(';')
        people_org_set.update(words)


def have_important_org_peo(sentence):
    for entity in people_org_set:
        if entity in sentence:
            return True
    return False


def get_entities_score(sentence):
    date_score = int(have_date(sentence))
    ple_org_score = int(have_important_org_peo(sentence))
    return 1.6 if (date_score + ple_org_score) > 0 else 1

def get_clue_score(sentences):
    clue_words = '总之 总而言之 综上 综上所述 一言以蔽之 概括起来说 括而言之 括而言之 要而论之 统而言之 归根到底 归根结底 简而言之'.split()
    result = []
    for sen in sentences:
        flag = 1
        for w in clue_words:
            if w in sen:
                flag = 1.4
                break
        result.append(flag)
    return np.array(result)
        
def auto_summary(doc,title,extract_num=None,title_common=False,use_mmr=True):
    sentences = split_document(doc)
    len_sen = len(sentences)
    if len_sen < 8:
        print(len_sen, end=';')
        return ''
    all_sentences_words = [split_func(sen) for sen in sentences]
    sentence_vec = get_sentnce_vector(all_sentences_words)
    pagerank_score = calc_page_rank(sentence_vec)
    entities_score = np.array([get_entities_score(sen) for sen in sentences])
    
    title_words = split_func(title)
    title_vec = get_sentnce_vector([title_words])
    title_common_score = get_title_common_score(all_sentences_words, title_words)
    title_sim_score = get_title_similarity(sentence_vec, title_vec)
    scaler = MinMaxScaler((1,2))
    scaler.fit(title_sim_score)

#     print(scaler.data_max_)
    print(title_sim_score.shape)
    title_sim_score = scaler.transform(title_sim_score)[:,0]
    print(title_sim_score.shape)
    position_score = get_position_score(len_sen)
    clue_score = get_clue_score(sentences)
    score = pagerank_score * entities_score * (title_common_score if title_common else title_sim_score) * position_score * clue_score
    if extract_num is None:
        extract_num = max(5, len_sen // 2)
        extract_num = min(extract_num, 23)
    #----------------MMR-------------------------------------
    n = extract_num
    summary_set = []
    alpha = 0.8
    max_score_index = np.argmax(score)
    summary_set.append(max_score_index)
    while n > 0:
        sim_mat = cosine_similarity(sentence_vec,sentence_vec[summary_set])
        sim_mat = np.max(sim_mat,axis=1)
        import pdb
#         pdb.set_trace()
        scaler = MinMaxScaler()
        feature_score = np.array([score,sim_mat]).T
        scaler.fit(feature_score)
        feature_score = scaler.transform(feature_score)
        [score,sim_mat] = feature_score[:,0], feature_score[:,1]
        mmr_score =  alpha*score - (1-alpha)*sim_mat
        mmr_score[summary_set] = -100
        max_index  = np.argmax(mmr_score)
        summary_set.append(max_index)
        n -= 1
    #----------------MMR-------------------------------------
    if not use_mmr:
        pagerank_sort = sorted(list(enumerate(score)), key=lambda x: x[1], reverse=True)[:extract_num]
        rank_keys = [k for k, v in pagerank_sort]
    else:
        rank_keys = summary_set
    color_list = ['firebrick','palevioletred','darkorchid','violet','chocolate','sandybrown','antuquewhite','darkkhaki','forestgreen','mediumblue']
    summary = ''.join([sen for idx, sen in enumerate(sentences) if idx in rank_keys])
    template = '<font style="background-color:{}" >{}</font>'
    template_normal = '<font  color="{}" >{}</font>'
    html_summary = ''.join([template.format('red',sen) if idx in rank_keys else template_normal.format(random.choice(color_list),sen)
                            for idx, sen in enumerate(sentences)])
    print('--',end=';')
    feature_df = pd.DataFrame({k:v for v,k in zip([score,pagerank_score,entities_score, title_sim_score, position_score,clue_score,sentences],
                              ['score','pagerank_score','entities_score', 'title_sim_score', 'position_score','clue_score','sentences'])}
                             )
    return summary, html_summary,(sentences, score),feature_df


content = """
新冠肺炎疫情暴发以来，频繁出现的无症状感染者病例，再次引起恐慌。近日，国家卫健委首度公布无症状感染者的情况。截至3月31日24时，31个省（自治区、直辖市）和新疆生产建设兵团报告新增无症状感染者130例，当日转为确诊病例2例，当日解除隔离302例。尚在医学观察无症状感染者1367例，比前一日减少174例。

那么，到底谁是无症状感染者？这些隐匿的感染者还有多少？会不会引爆第二波疫情？香港大学李嘉诚医学院教授高本恩告诉《中国科学报》：“第二波疫情是否到来关键看4月底，但无症状感染者不是主因。”

谁是“无症状感染者”？

3月31日，国务院新闻办召开的发布会上公布了无症状感染者的最新定义，即无发烧、咳嗽、咽痛等自我感知临床症状、无临床可识别症状体征，但呼吸道等样本病原学检测为阳性的患者。

对比国家卫健委3月7日在《新型冠状病毒肺炎防控方案（第六版）》中的定义，新定义增加了“自我感知”和“可识别症状”等主观感受方面的限定条件。过去一段时间，从新冠肺炎疫情发生初期，随着无症状感染者陆续在各地被通报，引发科技界高度关注和重视，并围绕其如何定义展开讨论。

1月29日，浙江杭州首次发现一名无症状感染者。《中国科学报》采访中，专家表示，这的确刷新了专业人士和公众认知。

中国工程院院士闻玉梅在当时的采访中强调了对无症状感染者的界定：“不发烧不等于没有症状，或者症状较轻容易被忽略。”“一定要非常慎重，不要因为误判引发恐慌。”

美国麻省大学医学院教授卢山也指出，证实感染者的确无症状，需要排除检测方法的假阳性、采集样本和检测中的交叉污染以及数据的可重复性等。

3月29日，国家卫健委专家组成员、北京地坛医院感染二科主任医师蒋荣猛在其个人微信公众号“北京也云感染”上发表文章称，即使是报告的“无症状感染者”，也可能存在因为症状轻微或不能正常主诉（如失语的老年人、儿童等）或因基础疾病如心血管疾病、慢性肺部疾病等症状的干扰导致信息采集偏离，同时也有客观证据显示部分“无症状感染者”其实有胸部X线检查异常表现。

3月31日，美国加州大学洛杉矶分校公共卫生学院副院长张作风接受《财经》采访时，仍然强调了排除主观因素重要性：病人在报告时可能会忽略胃痛、腹泻等症状，而这些有可能是感染新冠病毒的早期症状。

此前，一篇发表在《新英格兰医学杂志》上的论文就闹了“乌龙”。研究者报道了德国首次发现新冠病毒，病人是一位来自上海的当时无症状感染者。

几天后，研究者致函杂志，澄清了事实：作者在发表这篇论文之前并没有真正与这位女士沟通，信息仅来源于德国四位患者的口述，即“这位上海女同事似乎没有症状”。这名病人事实上出现了症状，她感到乏力、肌肉疼痛，并服用了退烧药扑热息痛。

新增限定条件围绕患者主观感受，回应了此前科学家们的担忧，让无症状感染者的统计在研究和防控方面更精准、更有针对性。

“冰山一角”将致第二波暴发？

2月5日，国家卫健委发布《新型冠状病毒感染的肺炎诊疗方案（试行第五版）》，首次提出“无症状感染者也可能成为传染源”。

蒋荣猛在公号文章中介绍：“从传染病的规律看，传染病流行通常有两个‘冰山’现象，即第一个冰山现象是感染后发病的是少数人，这也是为何要开展传染病报告、流行病学调查、密切接触者追踪的主要原因所在。第二个冰山现象是感染后发病人群中重症的比例占少数。”

在全国各地已经吹响复工复产号角的当下，人们担忧的是，“无症状感染者”会不会是第三个“冰山”——无症状感染者会不会在人群中占有不小的比例？他们携带病毒自由行动，正像隐匿的病毒传播者，最终导致疫情第二波暴发。

多项科学研究围绕这个问题展开。例如，美国乔治亚州立大学流行病学家Gerardo Chowell等学者3月曾在《欧洲监测》上发表研究，其对“钻石公主”号患者的模型统计显示，无症状患者比例为17.9%。对此，蒋荣猛在前述公号文章中指出，“钻石公主”号只是一个特例。

华中科技大学公共卫生学院教授邬堂春等学者，曾对武汉卫健委法定传染病报告系统中的确诊数据进行建模，得出武汉市至少有59%感染病例未被发现，其中包括无症状感染者和轻症患者。他在接受媒体采访时解释，该结果是基于“最保守的模型预测”，并未进行实地流行病学调查。

而据中国疾控中心2月17日在《中华流行病学》杂志上超7万人的大样本分析，889名无症状感染者占总数的1.2%。

中国工程院院士钟南山受访时，通过从结果反推的方式否定了无症状感染者“冰山一角”的担忧。他表示，无症状感染者对密切接触者传染率较高，而中国近期新增确诊病例数未升反降，据此可以推断，中国还没有大量的无症状感染者。

“历次疫情和疾病流行中都有无症状感染者出现，但这并非疫情再度暴发的诱因。”高本恩告诉《中国科学报》，“COVID-19最早在武汉出现是2019年12月初，大约1个月后才真正得到确认。其他国家的情况是，从2020年1月下旬输入性感染到2月下旬确认的社区感染，也大约是1个月。这样看来，未能严格控制境外输入病例、未能维持社区隔离才是可能导致疫情二次暴发的关键。

高本恩据此推测，当前措施的效果会在4月底前后显示出来。

浙江大学医学院公共卫生系教授金永堂告诉《中国科学报》：“没有证据表明我国存在二次暴发疫情和无症状感染者引发的疫情问题，否则我国本次疫情暴发与大流行不会如期得到顺利控制。”

（原标题为《 “无症状”恐引第二波疫情？专家表示不是主因》）
"""
title = '无症状感染者恐引第二波疫情？专家：不是主因，关键看4月底'



print(title)
num = 10
summary, html_summary,(sentences, score), feature_df= auto_summary(content, title,num,title_common=False,use_mmr=True)
display(HTML(html_summary))