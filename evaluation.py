import os
import os.path
import json
import numpy as np
import sys
sys.path.append('/kaggle/input/mydataset')
rel2id = json.load(open('/kaggle/input/mydataset/meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}


def is_intra_sentence(vertexSet, h_idx, t_idx):
    """
    判断头实体和尾实体是否在同一句话中出现过（句内关系）
    Args:
        vertexSet: 实体集合，每个实体包含多个 mentions，每个 mention 有 sent_id
        h_idx: 头实体索引
        t_idx: 尾实体索引
    Returns:
        True: 句内关系（头尾实体至少有一对 mention 在同一句中）
        False: 句间关系（头尾实体从未在同一句中共同出现）
    """
    h_sent_ids = set(mention['sent_id'] for mention in vertexSet[h_idx])
    t_sent_ids = set(mention['sent_id'] for mention in vertexSet[t_idx])
    # 如果有交集，说明存在同一句话中的 mention
    return len(h_sent_ids & t_sent_ids) > 0


def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path):
    '''
        Adapted from the official evaluation code
        Added Intra-sentence vs Inter-sentence F1 evaluation
    '''
    # truth_dir = os.path.join(path, 'ref') # path is /kaggle/input/... which is read-only
    truth_dir = os.path.join('/kaggle/working', 'ref') # Use writable working directory

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train.json"), truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}
    
    # 用于 Intra/Inter 分类的 Gold 标签统计
    std_intra = {}  # 句内关系的 Gold 标签
    std_inter = {}  # 句间关系的 Gold 标签
    
    # 用于 Evidence-1/2/3+ 分类的 Gold 标签统计
    std_evi1 = {}   # 1个证据句的关系
    std_evi2 = {}   # 2个证据句的关系
    std_evi3 = {}   # 3个及以上证据句的关系

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            
            # 判断是句内还是句间关系
            if is_intra_sentence(vertexSet, h_idx, t_idx):
                std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
            else:
                std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
            
            # 根据证据句数量分组
            evi_count = len(label['evidence'])
            if evi_count == 1:
                std_evi1[(title, r, h_idx, t_idx)] = set(label['evidence'])
            elif evi_count == 2:
                std_evi2[(title, r, h_idx, t_idx)] = set(label['evidence'])
            else:  # evi_count >= 3
                std_evi3[(title, r, h_idx, t_idx)] = set(label['evidence'])

    tot_relations = len(std)
    tot_intra_relations = len(std_intra)
    tot_inter_relations = len(std_inter)
    tot_evi1_relations = len(std_evi1)
    tot_evi2_relations = len(std_evi2)
    tot_evi3_relations = len(std_evi3)
    
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    # 整体统计
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0
    correct_in_train_annotated = 0
    
    # Intra/Inter 分类统计
    pred_intra = 0       # 预测的句内关系数量
    pred_inter = 0       # 预测的句间关系数量
    correct_intra = 0    # 正确预测的句内关系数量
    correct_inter = 0    # 正确预测的句间关系数量
    
    # Evidence-1/2/3+ 分类统计（只统计正确数量用于计算 Recall）
    correct_evi1 = 0     # 正确预测的1证据句关系数量
    correct_evi2 = 0     # 正确预测的2证据句关系数量
    correct_evi3 = 0     # 正确预测的3+证据句关系数量

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]
        
        # 判断当前预测是句内还是句间
        is_intra = is_intra_sentence(vertexSet, h_idx, t_idx)
        if is_intra:
            pred_intra += 1
        else:
            pred_inter += 1

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            
            # 统计 Intra/Inter 正确数量
            if is_intra:
                correct_intra += 1
            else:
                correct_inter += 1
            
            # 统计 Evidence-1/2/3+ 正确数量（根据 Gold 的证据数量分组）
            if (title, r, h_idx, t_idx) in std_evi1:
                correct_evi1 += 1
            elif (title, r, h_idx, t_idx) in std_evi2:
                correct_evi2 += 1
            elif (title, r, h_idx, t_idx) in std_evi3:
                correct_evi3 += 1
            
            in_train_annotated = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True

            if in_train_annotated:
                correct_in_train_annotated += 1

    # ========== 整体 F1 计算 ==========
    re_p = 1.0 * correct_re / len(submission_answer) if len(submission_answer) > 0 else 0
    re_r = 1.0 * correct_re / tot_relations if tot_relations > 0 else 0
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    # ========== Intra-sentence F1 计算 ==========
    intra_p = 1.0 * correct_intra / pred_intra if pred_intra > 0 else 0
    intra_r = 1.0 * correct_intra / tot_intra_relations if tot_intra_relations > 0 else 0
    if intra_p + intra_r == 0:
        intra_f1 = 0
    else:
        intra_f1 = 2.0 * intra_p * intra_r / (intra_p + intra_r)

    # ========== Inter-sentence F1 计算 ==========
    inter_p = 1.0 * correct_inter / pred_inter if pred_inter > 0 else 0
    inter_r = 1.0 * correct_inter / tot_inter_relations if tot_inter_relations > 0 else 0
    if inter_p + inter_r == 0:
        inter_f1 = 0
    else:
        inter_f1 = 2.0 * inter_p * inter_r / (inter_p + inter_r)

    # ========== Evidence-1/2/3+ Recall 计算 ==========
    evi1_r = 1.0 * correct_evi1 / tot_evi1_relations if tot_evi1_relations > 0 else 0
    evi2_r = 1.0 * correct_evi2 / tot_evi2_relations if tot_evi2_relations > 0 else 0
    evi3_r = 1.0 * correct_evi3 / tot_evi3_relations if tot_evi3_relations > 0 else 0

    # 打印详细的 Intra/Inter 统计信息
    print("\n" + "="*60)
    print("Intra-sentence vs Inter-sentence Evaluation Results")
    print("="*60)
    print(f"\n[Overall]")
    print(f"  Total Gold Relations: {tot_relations}")
    print(f"  Total Predictions: {len(submission_answer)}")
    print(f"  Correct: {correct_re}")
    print(f"  P: {re_p*100:.2f}% | R: {re_r*100:.2f}% | F1: {re_f1*100:.2f}%")
    
    print(f"\n[Intra-sentence] (entities appear in the same sentence)")
    print(f"  Gold: {tot_intra_relations} | Pred: {pred_intra} | Correct: {correct_intra}")
    print(f"  P: {intra_p*100:.2f}% | R: {intra_r*100:.2f}% | F1: {intra_f1*100:.2f}%")
    
    print(f"\n[Inter-sentence] (entities never appear in the same sentence)")
    print(f"  Gold: {tot_inter_relations} | Pred: {pred_inter} | Correct: {correct_inter}")
    print(f"  P: {inter_p*100:.2f}% | R: {inter_r*100:.2f}% | F1: {inter_f1*100:.2f}%")
    
    print("\n" + "="*60)
    print("Evidence Count Based Evaluation Results (Recall Only)")
    print("="*60)
    
    print(f"\n[Evidence-1] (relations supported by 1 evidence sentence)")
    print(f"  Gold: {tot_evi1_relations} | Correct: {correct_evi1} | Recall: {evi1_r*100:.2f}%")
    
    print(f"\n[Evidence-2] (relations supported by 2 evidence sentences)")
    print(f"  Gold: {tot_evi2_relations} | Correct: {correct_evi2} | Recall: {evi2_r*100:.2f}%")
    
    print(f"\n[Evidence-3+] (relations supported by 3+ evidence sentences)")
    print(f"  Gold: {tot_evi3_relations} | Correct: {correct_evi3} | Recall: {evi3_r*100:.2f}%")
    print("="*60 + "\n")

    # 返回结果，添加 intra/inter 和 evidence 的指标
    return {
        'overall': {
            'f1': re_f1,
            'p': re_p,
            'r': re_r,
            'evi_f1': evi_f1,
            'f1_ign': re_f1_ignore_train_annotated,
        },
        'intra': {
            'f1': intra_f1,
            'p': intra_p,
            'r': intra_r,
            'gold': tot_intra_relations,
            'pred': pred_intra,
            'correct': correct_intra,
        },
        'inter': {
            'f1': inter_f1,
            'p': inter_p,
            'r': inter_r,
            'gold': tot_inter_relations,
            'pred': pred_inter,
            'correct': correct_inter,
        },
        'evi1': {
            'r': evi1_r,
            'gold': tot_evi1_relations,
            'correct': correct_evi1,
        },
        'evi2': {
            'r': evi2_r,
            'gold': tot_evi2_relations,
            'correct': correct_evi2,
        },
        'evi3': {
            'r': evi3_r,
            'gold': tot_evi3_relations,
            'correct': correct_evi3,
        }
    }

