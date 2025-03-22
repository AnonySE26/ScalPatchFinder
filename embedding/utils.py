import os
import json
import multiprocessing
import tqdm

def get_commit_dict_train(repo2cve2negcommits, owner, repo, cve_list, BM25_K=10000):
    cve2commitlist = {}
    #print("place A", flush=True)
    for each_cve in cve_list:
        this_commit_list = get_commit_list_cve_train(repo2cve2negcommits, owner, repo, each_cve, BM25_K)
        cve2commitlist[each_cve] = this_commit_list
        
    return cve2commitlist

def get_commit_dict_test(feature_path, owner, repo, cve_list, BM25_K=10000):
    cve2commitlist = {}
    #print("place A", flush=True)
    for each_cve in cve_list:
        this_commit_list = get_commit_list_cve_test(feature_path, owner, repo, each_cve, BM25_K)
        cve2commitlist[each_cve] = this_commit_list
        
    return cve2commitlist    

def worker(args):
    feature_path, owner, repo, each_cve, BM25_K = args
    return get_commit_list_cve_test(feature_path, owner, repo, each_cve, BM25_K)

def get_commit_list_test(feature_path, owner, repo, cve_list, BM25_K=10000):
       # select top 10k commits
    commit_list = set([])
    #print("place A", flush=True)

    args_list = [(feature_path, owner, repo, each_cve, BM25_K) for each_cve in cve_list]
    
    with multiprocessing.Pool(processes=min(10, len(cve_list))) as pool:
        results = list(pool.imap_unordered(worker, args_list))

    for this_commit_list in results:
        commit_list.update(this_commit_list)
        
    return list(commit_list)

def get_commit_list_cve_train(repo2cve2negcommits, owner, repo, this_cve, BM25_K=10000):
    commit_list = set([])
    try:
        top_commit_list = repo2cve2negcommits[owner + "@@" + repo][this_cve]["top"][:BM25_K]
        random_commit_list = repo2cve2negcommits[owner + "@@" + repo][this_cve]["random"][:BM25_K]
        commit_list.update(top_commit_list)
        commit_list.update(random_commit_list)
        return commit_list
    except KeyError:
        print(f"KeyError: {owner}@@{repo} not found in repo2cve2negcommits")
        return set([])

def get_commit_list_train(repo2cve2negcommits, owner, repo, cve_list, BM25_K=10000):
    commit_list = set([])
    try:
        for each_cve in cve_list:
            this_commit_list = get_commit_list_cve_train(repo2cve2negcommits, owner, repo, each_cve, BM25_K)
            commit_list.update(this_commit_list)
        return commit_list
    except KeyError:
        print(f"KeyError: {owner}@@{repo} not found in repo2cve2negcommits")
        return set([])

def get_commit_list_cve_test(feature_path, owner, repo, this_cve, BM25_K=10000):
    file_path = feature_path + owner + "@@" + repo + "/bm25_time/result/" + this_cve + ".json"
    if not os.path.exists(file_path): return set([])
    commit_list = []
    commit2entry = json.load(open(file_path))
    #print(owner, repo)
    sorted_commit2entry = sorted(commit2entry.items(), key = lambda x:x[1].get("new_score", -1), reverse=True)
    for x in range(min(BM25_K, len(sorted_commit2entry))):
        this_commit = sorted_commit2entry[x][0]
        this_entry = sorted_commit2entry[x][1]
        if "new_score" in this_entry:
            commit_list.append(this_commit)
    return commit_list
