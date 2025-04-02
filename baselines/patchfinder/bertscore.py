import pandas as pd
from tqdm import trange

from bertscore_utils import *


def get_bert_embedding_optimized(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    batch_size=32,
    device="cuda:0",
    all_layers=False,
):
    """
    A simple optimization to get_bert_embedding that only computes the embeddings of the unique sentences
    """

    # Get unique sentences
    unique_sens = list(set(all_sens))

    # Tokenize and get IDF values over ALL sentences (including duplicates)
    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device
    )

    padded_unique_sens, _, _, unique_mask = collate_idf(
        unique_sens, tokenizer, idf_dict, device=device
    )

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(unique_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_unique_sens[i: i + batch_size],
                attention_mask=unique_mask[i: i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    unique_embeddings = torch.cat(embeddings, dim=0)

    sent_to_idx = {sent: idx for idx, sent in enumerate(unique_sens)}
    total_embedding = torch.stack([unique_embeddings[sent_to_idx[sent]] for sent in all_sens])

    return total_embedding, mask, padded_idf


##################################################
# MERGE TWO LOOPS INTO ONE
##################################################

def bert_cos_score_idf_optimized(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose=False,
    batch_size=64,
    device="cuda:0",
    all_layers=False,
):
    def pad_batch_stats(sentences, stats_dict, device):
        stats = [stats_dict[s] for s in sentences]
        emb, idf = zip(*stats)

        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0).to(device)
        idf_pad = pad_sequence(idf, batch_first=True).to(device)

        lens = torch.tensor([e.size(0) for e in emb], device=device)
        max_len = emb_pad.size(1)
        pad_mask = torch.arange(max_len, device=device).expand(len(lens), max_len) < lens.unsqueeze(1)

        return emb_pad, pad_mask, idf_pad


    #################################################

    assert len(refs) == len(hyps)
    preds = list()

    for batch_start in trange(0, len(refs), batch_size, desc="Computing Embeddings...", disable=not verbose):
        hyp_batch = hyps[batch_start: batch_start + batch_size]
        ref_batch = refs[batch_start: batch_start + batch_size]

        unique_sentences = list(set(hyp_batch + ref_batch))
        embs, masks, padded_idf = get_bert_embedding_optimized(
            unique_sentences, model, tokenizer, idf_dict, batch_size=batch_size, device=device, all_layers=all_layers
        )

        embs = embs.to(device)
        padded_idf = padded_idf.to(device)
        masks = masks.to(device)

        sequence_lengths = masks.sum(dim=1)
        stats_dict = {
            unique_sentences[i]: (embs[i, :sequence_lengths[i]], padded_idf[i, :sequence_lengths[i]])
            for i in range(len(unique_sentences))
        }

        del embs, masks, padded_idf
        torch.cuda.empty_cache()

        with torch.no_grad():
            ref_stats = pad_batch_stats(ref_batch, stats_dict, device)
            hyp_stats = pad_batch_stats(hyp_batch, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())

        del stats_dict
        torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


##################################################
def score_optimized(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
):
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert (
        lang is not None or model_type is not None
    ), "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()

    ####################################################################################################
    # core code
    # the main bottleneck is the get_bert_embedding used by bert_cos_score_idf

    all_preds = bert_cos_score_idf_optimized(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    ####################################################################################################
    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(
                    pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                )[1:].float()
            else:
                baselines = (
                    torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:]
                    .unsqueeze(1)
                    .float()
                )

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(
            f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
        )

    if return_hash:
        return tuple(
            [
                out,
                get_hash(
                    model_type,
                    num_layers,
                    idf,
                    rescale_with_baseline,
                    use_custom_baseline=use_custom_baseline,
                    use_fast_tokenizer=use_fast_tokenizer,
                ),
            ]
        )

    return out