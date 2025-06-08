import glob, json
import os.path as osp

def count_vlm_results(keys, name_even):
    fname = 'object_IQA-quality-1+3-new-ref'
    model_name = 'InternVL2_5-78B'
    json_files = sorted(glob.glob(osp.join(name_even, f'*/{fname}_{model_name}.json')))
    # print(json_files)
    counts = {}
    total = 0
    for json_file in json_files:
        d = json.load(open(json_file))
        total += len(d.keys())
        for k, v in d.items():
            if k == 'prompts':
                total -= 1
                continue
            for idx, ans in enumerate(v):
                if keys[idx] not in counts:
                    counts[keys[idx]] = 0
                if 'yes' in ans.lower():
                    counts[keys[idx]] += 1
    counts['total'] = total
    return counts