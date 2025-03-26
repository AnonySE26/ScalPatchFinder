from pandas import read_csv
import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
sys.path.append("../")
import os
from recall import get_recall_from_sortedcommits
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory
from utils import get_commit_dict_test, get_commit_dict_train
import pandas
import argparse

global_embeddings_shm = None
global_embeddings_shape = None
global_embeddings_dtype = None



def init_worker_file(emb_shm_name, emb_shape, emb_dtype, commit_shm_name, commit_shape, commit_dtype, fileid_shm_name, fileid_shape, fileid_dtype):
    global global_embeddings_shm, global_embeddings_shape, global_embeddings_dtype, global_commit_shm, global_commit_shape, global_commit_dtype, global_fileid_shm, global_fileid_shape, global_fileid_dtype
    global_embeddings_shm = shared_memory.SharedMemory(name=emb_shm_name)
    global_embeddings_shape = emb_shape
    global_embeddings_dtype = emb_dtype

    global_commit_shm = shared_memory.SharedMemory(name=commit_shm_name)
    global_commit_shape = commit_shape
    global_commit_dtype = commit_dtype

    global_fileid_shm = shared_memory.SharedMemory(name=fileid_shm_name)
    global_fileid_shape = fileid_shape
    global_fileid_dtype = fileid_dtype

def init_worker_notfile(emb_shm_name, emb_shape, emb_dtype, commit_shm_name, commit_shape, commit_dtype):
    global global_embeddings_shm, global_embeddings_shape, global_embeddings_dtype, global_commit_shm, global_commit_shape, global_commit_dtype
    global_embeddings_shm = shared_memory.SharedMemory(name=emb_shm_name)
    global_embeddings_shape = emb_shape
    global_embeddings_dtype = emb_dtype

    global_commit_shm = shared_memory.SharedMemory(name=commit_shm_name)
    global_commit_shape = commit_shape
    global_commit_dtype = commit_dtype
    
def get_gt_commits(data):
    repo2cve2gtcommits = {}
    for idx in range(len(data)):
        cve = data["cve"].iloc[idx]
        owner = data["owner"].iloc[idx]
        repo = data["repo"].iloc[idx]
        owner_repo = owner + "@@" + repo
        commit = data["patch"].iloc[idx]
        repo2cve2gtcommits.setdefault(owner_repo, {})
        repo2cve2gtcommits[owner_repo].setdefault(cve, [])
        repo2cve2gtcommits[owner_repo][cve].append(commit)
    return repo2cve2gtcommits

def cachenvd():
    cve2desc = {}
    for year in range(2002, 2025):
        print(year)
        data = json.load(open(f"../../NVD/nvdcve-1.1-{year}.json", "r"))
        for idx in range(len(data["CVE_Items"])):
            cve = data["CVE_Items"][idx]["cve"]["CVE_data_meta"]["ID"]
            description = data["CVE_Items"][idx]["cve"]["description"]["description_data"]
            cve2desc[cve] = description
    json.dump(cve2desc, open("../../NVD/cve2desc.json", "w"))

def get_ranked_list(query_vector, embeddings):
    query_vector = np.array(query_vector).reshape(1, -1)
    doc_vectors = np.array(embeddings)
    return cosine_similarity(query_vector, doc_vectors)

def get_cosine_sim(owner, repo, each_cve, query_embedding, all_embeddings, all_commits, all_fileids = None, is_file = False, head_count=2, mode = "mean"):

    if is_file is False:
        all_cosine = cosine_similarity(np.array(query_embedding).reshape(1, -1), all_embeddings)[0]
        return all_cosine, all_commits
    else:
        # bind the dictionary all_embeddings and commit2bm25score_file with key 
        print("reading", feature_path + owner + "@@" + repo_name + "/bm25_files/result/" + each_cve + ".csv")
        try:
            commit2bm25score_file = read_csv(feature_path + owner + "@@" + repo_name + "/bm25_files/result/" + each_cve + ".csv", sep=",", on_bad_lines='skip', dtype={"commit_id": str, "filename":str, "bm25score": "float64", "file_id": "Int64"}).drop_duplicates(subset=["commit_id", "file_id"])
        except Exception as e:
            print(e)
            return None, None
        print("before", len(commit2bm25score_file))

        commit_file_list = pandas.DataFrame({"commit_id": all_commits, "file_id": all_fileids})

        commit2bm25score_file = commit_file_list.merge(commit2bm25score_file, on=["commit_id", "file_id"], how="left").fillna(-1)

        commit2bm25score_file.reset_index(drop=True, inplace=True)
        commit2bm25score_file["origin_index"] = commit2bm25score_file.index
        sorted_commit2bm25score_file = commit2bm25score_file.sort_values(["commit_id", "bm25score"], ascending=[True, False])
        top_k_per_group = sorted_commit2bm25score_file.groupby("commit_id")

        final_commit_id = list(top_k_per_group.groups.keys())
        
        top_k_per_group = top_k_per_group.head(head_count)
        group_sizes = top_k_per_group.groupby("commit_id").size().reset_index(name='group_size')

        reordered_embeddings = all_embeddings[top_k_per_group["origin_index"]]

        group_sizes_array = group_sizes["group_size"].values
        group_start_indices = np.concatenate(([0], np.cumsum(group_sizes_array)[:-1]))

        if mode == "mean":
            group_embedding_sum = np.add.reduceat(reordered_embeddings, group_start_indices, axis=0)
            group_sizes_reshaped = group_sizes_array[:, np.newaxis] 

            group_embeddings_avg = np.array(group_embedding_sum / group_sizes_reshaped)

            all_cosine = cosine_similarity(np.array(query_embedding).reshape(1, -1), group_embeddings_avg)[0]

        elif mode == "max":
            all_cosine_reordered = cosine_similarity(np.array(query_embedding).reshape(1, -1), reordered_embeddings)[0]
            all_cosine = np.maximum.reduceat(all_cosine_reordered, group_start_indices, axis=0)

        #raise Exception(len(all_max_cosine), len(all_cosine_reordered), len(all_cosine)) #, len(final_commit_id))

        return all_cosine, final_commit_id
    
def process_single_cve_notfile(each_row):

    each_cve, this_candidate_commits, gt_commits, query_embedding, output_dir = each_row

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{each_cve}.json")

    #query_embedding = cve2embedding[each_cve]
    # use shared embeddings
    all_embeddings = np.ndarray(global_embeddings_shape, dtype=global_embeddings_dtype, buffer=global_embeddings_shm.buf)
    all_commits = np.ndarray(global_commit_shape, dtype=global_commit_dtype, buffer=global_commit_shm.buf)

    
    if len(set(gt_commits).intersection(this_candidate_commits)) == 0:
        print(f"CVE: {each_cve}, ground truth commits: {len(gt_commits)}, matched in commits: {len(set(gt_commits).intersection(this_candidate_commits))}", flush=True)

        if os.path.exists(output_file):
            os.remove(output_file)
        return False # {}
    
    all_cosine = cosine_similarity(np.array(query_embedding).reshape(1, -1), all_embeddings)[0]

    cosine_commit = {all_commits[x]: all_cosine[x] for x in range(len(all_cosine)) if all_commits[x] in this_candidate_commits}
    sorted_cosine_commit = sorted(cosine_commit.items(), key=lambda x: x[1], reverse=True)
    
    print("dumping", output_file)
    with open(output_file, "w") as f:
        output_list = {commit_id: {"new_score": score} for (commit_id, score) in sorted_cosine_commit}
        json.dump(output_list, f, indent=4)

    #sorted_commits = [x[1] for x in sorted_cosine_commit]

    return True# get_recall_from_sortedcommits(each_cve, sorted_commits, gt_commits)

def process_single_cve_file(each_row):

    each_cve, owner, repo, this_candidate_commits, gt_commits, query_embedding, output_dir, is_file, each_suffix, head_count, mode = each_row

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if os.path.exists(output_dir + f"{each_cve}.json"):
    #     print(output_dir + f"{each_cve}.json already exists")
    #     return False
    output_file = os.path.join(output_dir, f"{each_cve}_{each_suffix}.json")

    all_embeddings = np.ndarray(global_embeddings_shape, dtype=global_embeddings_dtype, buffer=global_embeddings_shm.buf)
    all_commits = np.ndarray(global_commit_shape, dtype=global_commit_dtype, buffer=global_commit_shm.buf)
    all_fileids = np.ndarray(global_fileid_shape, dtype=global_fileid_dtype, buffer=global_fileid_shm.buf)
    
    if len(set(gt_commits).intersection(this_candidate_commits)) == 0:
        print(f"CVE: {each_cve}, ground truth commits: {len(gt_commits)}, matched in commits: {len(set(gt_commits).intersection(this_candidate_commits))}", flush=True)

        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            return False # {}
        except FileNotFoundError:
            return False
    
    if not is_file:
        all_cosine, all_commits = get_cosine_sim(owner, repo, each_cve, query_embedding, all_embeddings, all_commits)
    else:
        all_cosine, all_commits = get_cosine_sim(owner, repo, each_cve, query_embedding, all_embeddings, all_commits, all_fileids=all_fileids, is_file=is_file, head_count=head_count, mode = mode)

    if all_cosine is None or all_commits is None:
        print(f"Error in processing {each_cve} for {owner}/{repo}")
        return False

    assert len(all_cosine) == len(all_commits)

    cosine_commit = {all_commits[x]: all_cosine[x] for x in range(len(all_cosine)) if all_commits[x] in this_candidate_commits}

    print("dumping", output_file)
    with open(output_file, "w") as f:
        output_list = {commit_id: {"new_score": score} for (commit_id, score) in cosine_commit.items()}
        json.dump(output_list, f, indent=4)

    #sorted_commits = [x[1] for x in sorted_cosine_commit]

    return True# get_recall_from_sortedcommits(each_cve, sorted_commits, gt_commits)

def reduce_result(cve_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import glob

    for each_cve in list(set(cve_list)):
        file_list = glob.glob(f"{output_dir}/{each_cve}_*.json")

        if len(file_list) == 0:
            continue

        merged_dict = {}
        for each_file in file_list:
            with open(each_file, "r") as f:
                data = json.load(f)
                merged_dict.update(data)
        
        with open(f"{output_dir}/{each_cve}.json", "w") as f:
            json.dump(merged_dict, f, indent=4)
        
        # for each_file in file_list:
        #     os.remove(each_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="grit_instruct_512_file")
    parser.add_argument("--dataset_name", type=str, default="AD")
    parser.add_argument("--is_file", action="store_true")
    parser.add_argument("--head_count", type=int, default=2)
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--mode", type=str, default="mean")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    is_file = args.is_file

    head_count = args.head_count
    is_train = args.is_train
    mode = args.mode

    fold = "train" if is_train else "test"

    repo_directory = "../"
    feature_path = "../feature/"

    data = read_csv(f"../csv/{dataset_name}_{fold}.csv")
    groupby_list = sorted(list(data.groupby(["owner", "repo"])), key = lambda x:x[0])

    import glob

    repo2cve2gitcommits = get_gt_commits(data)

    repo2cve2negcommits = json.load(open(f"../feature/repo2cve2negcommits_{dataset_name}_500_unsampled.json" if dataset_name == "patchfinder" else f"../feature/repo2cve2negcommits_{dataset_name}_500.json", "r"))
                                         
    #remaining_repos = read_csv("tmp/missed.txt")
    #remaining_repos = set([(remaining_repos.iloc[x]["owner"], remaining_repos.iloc[x]["repo"]) for x in range(len(remaining_repos))])
    # #raise Exception(remaining_repos)

    train_K = 500
    after_snapcore = False

    repo_count = 0

    suffix = "result_head" + str(head_count) if mode == "mean" else "result_head" + str(head_count) + "_max"

    if not is_file:
        suffix = "result"
    #remaining_repos = {'apache@@ozone', 'github@@hubot-scripts', 'element-hq@@synapse', 'FasterXML@@jackson-dataformats-text', 'apache@@servicecomb-java-chassis', 'h2database@@h2database', 'jooby-project@@jooby', 'evershopcommerce@@evershop', 'aws@@aws-cdk', 'pnpm@@pnpm', 'filebrowser@@filebrowser', 'kubernetes@@ingress-nginx', 'codenameone@@CodenameOne', 'node-red@@node-red', 'mapproxy@@mapproxy', 'stanfordnlp@@CoreNLP', 'felixrieseberg@@windows-build-tools', 'openshift@@cluster-monitoring-operator', 'wanasit@@chrono', 'apache@@incubator-dolphinscheduler', 'knative@@serving', 'labring@@sealos', 'csaf-poc@@csaf_distribution', 'SiCKRAGE@@SiCKRAGE', 'OHDSI@@WebAPI', 'ansible@@ansible-runner', 'theupdateframework@@tuf', 'kubernetes@@apiextensions-apiserver', 'Azure@@aad-pod-identity', 'swagger-api@@swagger-codegen', 'getsentry@@sentry-javascript', 'localstack@@localstack', 'wger-project@@wger', 'TeraTermProject@@teraterm', 'hyperledger@@aries-cloudagent-python', 'mongodb@@mongo-tools', 'mjwwit@@node-XMLHttpRequest', 'HtmlUnit@@htmlunit', 'plotly@@dash', 'kubernetes-sigs@@secrets-store-csi-driver', 'git-lfs@@git-lfs', 'stleary@@JSON-java', 'browserless@@chrome', 'pingcap@@tidb', 'hibernate@@hibernate-orm', 'hornetq@@hornetq', 'richfaces@@richfaces', 'significant-gravitas@@autogpt', 'nwjs@@npm-installer', 'transifex@@transifex-client', 'nicolargo@@glances', 'graphhopper@@graphhopper', 'multiversx@@mx-chain-go', 'nuxt@@nuxt', 'apache@@lucene', 'celery@@celery'}

    for (owner, repo_name), each_row in tqdm.tqdm(groupby_list):
        if (owner, repo_name) != ("xuxueli", "xxl-job"): continue
        ##if owner + "@@" + repo_name not in repo2cve2negcommits: continue 
        #if dataset_name == "patchfinder":
        #   if (owner, repo_name) == ("snapcore", "snapd"): 
        #       after_snapcore = True
        #       continue
        #   #if not after_snapcore: continue
        #2if is_train and (owner + "@@" + repo_name not in repo2cve2negcommits): continue
        cve_list = each_row["cve"].tolist()
        if is_file:
            if not (os.path.exists(feature_path + owner + "@@" + repo_name + f"/{model_name}/cve2embedding.json") and glob.glob(feature_path + owner + "@@" + repo_name + f"/{model_name}/commit_list_*.csv")):
                print(owner, repo_name, "not exist")
                continue
            if os.path.exists(repo_directory + f"./feature/{owner}@@{repo_name}/{model_name}/{suffix}/") and len(glob.glob(repo_directory + f"./feature/{owner}@@{repo_name}/{model_name}/{suffix}/*_0.json")) >= len(set(cve_list)):
                print(owner, repo_name, "already exist")
                continue
            output_dir = repo_directory + f"./feature/{owner}@@{repo_name}/{model_name}/{suffix}/"
        else:
            if not os.path.exists(feature_path + owner + "@@" + repo_name + f"/{model_name}/commit2embedding.json"):
                print(owner, repo_name, "not exist")
                continue
            output_dir = repo_directory + f"./feature/{owner}@@{repo_name}/{model_name}/{suffix}/"
        #if owner + "@@" + repo_name not in remaining_repos: continue
        
        os.makedirs(output_dir, exist_ok=True)

        if is_train is False:
            cve2commitlist = get_commit_dict_test(feature_path, owner, repo_name, cve_list, BM25_K=10000)
        else:
            cve2commitlist = get_commit_dict_train(repo2cve2negcommits, owner, repo_name, cve_list, BM25_K=train_K)
            for each_cve in cve2commitlist:
                cve2commitlist[each_cve].update(repo2cve2gitcommits[owner + "@@" + repo_name][each_cve])


        cve2gtcommits = repo2cve2gitcommits[owner + "@@" + repo_name]
        
        print("finish loading cve2commit")

        if model_name == "path_":
            cve2embedding = json.load(open(repo_directory + "/feature/" + owner + "@@" + repo_name + "/voyage/cve2embedding.json", "r"))
        else:
            print("cve2embedding", repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + "/cve2embedding.json")
            cve2embedding = json.load(open(repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + "/cve2embedding.json", "r"))
        # commit2embedding = json.load(open(repo_directory + "/commit2embedding.json", "r"))
        # cve2embedding = json.load(open("./cve2embedding2.json", "r"))

        if is_file:
            filelist = glob.glob(repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + "/commit_list_*.csv")

            for each_file in filelist:
                each_suffix = each_file.split("_")[-1][:-4]
                print("suffix", each_suffix)
            
                commit_and_filename = read_csv(repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + f"/commit_list_{each_suffix}.csv", dtype={"commit_id": str, "filename":str,"original_index": "Int64", "file_id": "Int64"})
                print("embeddin file:", repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + f"/embedding_list_{each_suffix}.json")
                embeddings_np = np.array(json.load(open(repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + f"/embedding_list_{each_suffix}.json", "r")))

                print("finish loading embedding")
                
                try:
                    emb_shm = shared_memory.SharedMemory(create=True, size=embeddings_np.nbytes, name="embeddings_shm")
                    shared_embeddings = np.ndarray(embeddings_np.shape, dtype=embeddings_np.dtype, buffer=emb_shm.buf)
                    shared_embeddings[:] = embeddings_np[:]
                    
                    commit_list = np.array(list(commit_and_filename["commit_id"]))
                    commit_shm = shared_memory.SharedMemory(create=True, size=commit_list.nbytes, name="commits_shm")
                    shared_commits = np.ndarray(commit_list.shape, dtype=commit_list.dtype, buffer=commit_shm.buf)
                    shared_commits[:] = commit_list[:] 

                    fileid_list = np.array(commit_and_filename["file_id"])
                    fileid_shm = shared_memory.SharedMemory(create=True, size=fileid_list.nbytes, name="fileid_shm")
                    shared_fileid = np.ndarray(fileid_list.shape, dtype=fileid_list.dtype, buffer=fileid_shm.buf)
                    shared_fileid[:] = fileid_list[:]
                except FileExistsError:
                    emb_shm.close()
                    emb_shm.unlink()

                    commit_shm.close()
                    commit_shm.unlink()

                    fileid_shm.close()
                    fileid_shm.unlink()
                    continue
                
                try:
                    all_examples = [(cve, owner, repo_name, cve2commitlist[cve], cve2gtcommits[cve], cve2embedding[cve], output_dir, is_file, each_suffix, head_count, mode) for cve in cve_list if not os.path.exists(output_dir + f"{cve}.json")]
                except KeyError as e:
                    print(f"KeyError: {e} not found in cve2commitlist")
                    continue

                results = []
                with Pool(processes=10, initializer=init_worker_file, initargs=(emb_shm.name, embeddings_np.shape, embeddings_np.dtype, commit_shm.name, commit_list.shape, commit_list.dtype, fileid_shm.name, fileid_list.shape, fileid_list.dtype)) as pool:
                    for res in tqdm.tqdm(pool.imap_unordered(process_single_cve_file, all_examples), total=len(all_examples)):
                        pass

                emb_shm.close()
                emb_shm.unlink()

                commit_shm.close()
                commit_shm.unlink()

                fileid_shm.close()
                fileid_shm.unlink()

            if is_file:
                reduce_result(cve_list, output_dir)
        else:
            commit2embedding = json.load(open(repo_directory + "/feature/" + owner + "@@" + repo_name + "/" + model_name + "/commit2embedding.json", "r"))
            commits = list(commit2embedding.keys())
            # embeddings = list(commit2embedding.values())
            # print(len(embeddings))
            
            embedding_list = list(commit2embedding.values())
            embeddings_np = np.array(embedding_list)
            if embeddings_np.nbytes == 0:
                continue
            emb_shape = embeddings_np.shape
            emb_dtype = embeddings_np.dtype

            
            emb_shm = shared_memory.SharedMemory(create=True, size=embeddings_np.nbytes, name="embeddings_shm")
            shared_embeddings = np.ndarray(embeddings_np.shape, dtype=embeddings_np.dtype, buffer=emb_shm.buf)
            shared_embeddings[:] = embeddings_np[:]
            
            commit_list = np.array(list(commit2embedding.keys()))
            commit_shm = shared_memory.SharedMemory(create=True, size=commit_list.nbytes, name="commits_shm")
            shared_commits = np.ndarray(commit_list.shape, dtype=commit_list.dtype, buffer=commit_shm.buf)
            shared_commits[:] = commit_list[:] 
            
            output_dir = repo_directory + f"./feature/{owner}@@{repo_name}/{model_name}/result/"
            os.makedirs(output_dir, exist_ok=True) 
            
            all_examples = []
            for cve in cve_list:
                this_tuple = (cve, cve2commitlist.get(cve, None), cve2gtcommits.get(cve, None), cve2embedding.get(cve, None), output_dir)
                if this_tuple[3] and this_tuple[1] and this_tuple[2]:
                    all_examples.append(this_tuple)

            results = []
            with Pool(processes=10, initializer=init_worker_notfile, initargs=(emb_shm.name, emb_shape, emb_dtype, commit_shm.name, commit_list.shape, commit_list.dtype)) as pool:
                for res in tqdm.tqdm(pool.imap_unordered(process_single_cve_notfile, all_examples), total=len(all_examples)):
                    pass

            emb_shm.close()
            emb_shm.unlink()

            commit_shm.close()
            commit_shm.unlink()
