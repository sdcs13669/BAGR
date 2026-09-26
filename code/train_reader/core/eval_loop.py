"""Evaluation loops: evidence reader + fixed strategies.

JSON schema is stable across versions. The evidence reader rolls out via
forward_states_evidence (selection identical to the base pipeline); its
terminal judgment is the argmax of ev_logits from one extra evidence forward
after the budget is exhausted, and the frozen upper-bound classifier on the
same final subgraphs is recorded as the acc_clf_* reference. The agentnet
baseline carries its own readout head: protocol judgment stays the frozen
classifier, and its own-head judgment is additionally recorded as acc_own_*.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed


@torch.no_grad()
def rollout_strategy(agent, state):
    """Roll out one graph: agent.select until the budget is exhausted

    or the frontier is empty.

    Stateful strategies (e.g. the BFS FIFO queue) rely on the
    reset/on_new_frontier hooks: each rollout resets internal state and
    seeds it with the initial frontier.
    """
    agent.reset()
    initial_frontier = list(state._initial_frontier)
    if initial_frontier:
        agent.on_new_frontier(initial_frontier)
    while not state.is_frontier_empty and state.get_budget_used() < state._budget_limit:
        v = agent.select(state)
        if v is None:
            break
        new_frontier = reveal_node(state, v)
        if new_frontier:
            agent.on_new_frontier(new_frontier)


def _edge_key_to_id(g) -> dict:
    """canonical (min,max) key -> first-seen edge id (matches GraphState).



    Prebuilt once per graph and shared across all its rollouts.
    """
    if 'feat' not in g.edata:
        return {}
    src, dst = g.edges()
    out = {}
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in out:
            out[key] = eid
    return out


@torch.no_grad()
def evaluate_reader(reader, classifier, builder, graphs, labels,
                    budget_options, device, argmax=True, chunk_size=32,
                    n_seeds=4, confidence_out=None):
    """Reader evaluation: cross-graph lockstep, batched classifier,

    multi-budget / multi-seed statistics.

    Evidence reader: terminal judgment comes from its own evidence head
    (one extra forward after the budget is exhausted); the frozen
    upper-bound classifier on the same revealed subgraphs is recorded as
    the acc_clf_* reference.

    When confidence_out is not None (calibration analysis), collect
    per-budget samples [max class confidence (4-dp), correct]: ev =
    evidence-head judgment (evidence reader only),
    clf = frozen classifier on the same revealed subgraph.
    """
    reader.eval()
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}
    per_budget_clf_correct = defaultdict(int)
    per_budget_clf_outcomes = defaultdict(list)
    use_evidence = (getattr(reader, 'config', None) is not None
                    and getattr(reader.config, 'kind', '') == 'evidence')

    for chunk_start in tqdm(range(0, len(graphs), chunk_size), desc="  Revealing"):
        chunk_end = min(chunk_start + chunk_size, len(graphs))
        all_states, state_info = [], []
        for local_i in range(chunk_start, chunk_end):
            g_raw = graphs[local_i]
            label = int(labels[local_i].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()
            ek2id = _edge_key_to_id(g_raw)
            for budget_ratio in budget_options:
                B = max(1, int(budget_ratio * n_nodes))
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node,
                                       edge_key_to_id=ek2id)
                    state._budget_limit = B
                    all_states.append(state)
                    state_info.append((y, label, budget_ratio, seed_i))

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)
        if use_evidence:
            all_hc = reader.init_evidence(n_total, device)
            last_revealed = [s.seed_node for s in all_states]

        state_rng = [random.Random(s.seed_node * 1_000_003 + s._budget_limit)
                     for s in all_states]
        for _step in range(max_budget):
            for j in range(n_total):
                if active[j] and all_states[j].is_frontier_empty:
                    st = all_states[j]
                    tp_node = teleport_if_stuck(st, rng=state_rng[j])
                    if tp_node is not None:
                        if use_evidence:
                            last_revealed[j] = tp_node
                        if st.get_budget_used() >= st._budget_limit:
                            active[j] = False
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break
            if use_evidence:
                batch_results, active_hc, _ = reader.forward_states_evidence(
                    [all_states[j] for j in active_idx], temperature=1.0,
                    hc_list=[all_hc[j] for j in active_idx],
                    last_revealed=[last_revealed[j] for j in active_idx])
                for k, j in enumerate(active_idx):
                    all_hc[j] = active_hc[k]
            else:
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx], temperature=1.0)

            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                chosen = frontier_nodes[probs.argmax().item()] if argmax else \
                    frontier_nodes[torch.distributions.Categorical(probs).sample().item()]
                reveal_node(all_states[j], chosen)
                if use_evidence:
                    last_revealed[j] = chosen
            for j in active_idx:
                if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        if use_evidence:
            _, _, final_ev = reader.forward_states_evidence(
                all_states, temperature=1.0, hc_list=all_hc,
                last_revealed=last_revealed)

        all_subgraphs_out, all_labels_out, all_budgets_out, all_seed_ids_out = \
            [], [], [], []
        for j in range(n_total):
            sub_g, _ = builder.build(all_states[j])
            all_subgraphs_out.append(sub_g)
            all_labels_out.append(state_info[j][:2])
            all_budgets_out.append(state_info[j][2])
            all_seed_ids_out.append(state_info[j][3])

        for i in range(0, len(all_subgraphs_out), 512):
            chunk_sub = all_subgraphs_out[i:i + 512]
            batch_g = dgl.batch(chunk_sub).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(batch_g, nids)
            for j, (y, label) in enumerate(all_labels_out[i:i + 512]):
                budget = all_budgets_out[i + j]
                seed_i = all_seed_ids_out[i + j]
                if use_evidence:
                    correct = 1.0 if final_ev[i + j].argmax(dim=-1).item() == label else 0.0
                    clf_correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
                    per_budget_clf_correct[budget] += clf_correct
                    per_budget_clf_outcomes[budget].append(clf_correct)
                else:
                    correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
                if confidence_out is not None:
                    rec = confidence_out.setdefault(
                        budget, {'ev': [], 'clf': []})
                    if use_evidence:
                        p_ev = torch.softmax(final_ev[i + j].float(), dim=-1)
                        rec['ev'].append(
                            [round(float(p_ev.max()), 4), int(correct)])
                    p_clf = torch.softmax(logits[j].float(), dim=-1)
                    rec['clf'].append(
                        [round(float(p_clf.max()), 4),
                         int(clf_correct if use_evidence else correct)])
                per_budget_correct[budget] += correct
                per_budget_total[budget] += 1
                per_budget_outcomes[budget].append(correct)
                per_seed_outcomes[seed_i][budget].append(correct)
                per_seed_per_class[seed_i][budget][label].append(correct)

    return _summarize(budget_options, graphs, per_budget_correct,
                      per_budget_total, per_budget_outcomes,
                      per_seed_outcomes, per_seed_per_class,
                      clf_correct=(per_budget_clf_correct if use_evidence else None),
                      clf_outcomes=(per_budget_clf_outcomes if use_evidence else None))


@torch.no_grad()
def evaluate_fixed(strategy, classifier, builder, graphs, labels,
                   budget_options, device, argmax=True, n_seeds=4, agent=None,
                   judge='clf'):
    """Fixed-strategy (random/bfs/max_degree) evaluation; statistics match
    evaluate_reader.


    An externally constructed strategy instance (protocol-adapted baseline
    such as sagpool, which needs a scorer checkpoint) may be passed;
    strategy is then only a log/registry name and may be None.

    judge='clf' (default): frozen upper-bound classifier is the terminal
    judge; own-head strategies additionally record acc_own_* (reference).
    judge='own': the strategy's own head is the terminal judge (acc_*),
    and the frozen classifier on the same subgraphs is recorded as the
    acc_clf_* reference. Requires the agent to implement own_predict.
    """
    from active_graph_reader.reader.fixed import create_fixed_reader
    if agent is None:
        agent = create_fixed_reader(strategy)
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}
    own_fn = getattr(agent, 'own_predict', None)
    own_batch_fn = getattr(agent, 'own_predict_batch', None)
    if judge == 'own' and own_fn is None and own_batch_fn is None:
        raise ValueError(f"strategy {strategy or getattr(agent, 'name', '?')} "
                         "has no own head (own_predict/own_predict_batch); "
                         "judge='own' is unsupported")
    own_any = own_fn is not None or own_batch_fn is not None
    per_budget_own_correct = defaultdict(int)
    per_budget_own_outcomes = defaultdict(list)
    per_budget_clf_correct = defaultdict(int)
    per_budget_clf_outcomes = defaultdict(list)
    all_subgraphs, all_labels, all_budgets, all_seed_ids = [], [], [], []
    all_seed_rows, all_steps = [], []
    for local_i in tqdm(range(len(graphs)), desc="  Rolling"):
        g_raw = graphs[local_i]
        label = int(labels[local_i].item())
        y = torch.tensor([label], device=device)
        n_nodes = g_raw.num_nodes()
        ek2id = _edge_key_to_id(g_raw)
        for budget_ratio in budget_options:
            B = max(1, int(budget_ratio * n_nodes))
            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node,
                                   edge_key_to_id=ek2id)
                state._budget_limit = B
                rollout_strategy(agent, state)
                sub_g, sub_ids = builder.build(state)
                all_subgraphs.append(sub_g)
                all_labels.append((y, label))
                all_budgets.append(budget_ratio)
                all_seed_ids.append(seed_i)
                all_seed_rows.append(sub_ids.index(state.seed_node))
                all_steps.append(B)
    for i in range(0, len(all_subgraphs), 512):
        chunk_sub = all_subgraphs[i:i + 512]
        batch_g = dgl.batch(chunk_sub).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)
        own_logits = (own_batch_fn(chunk_sub, all_seed_rows[i:i + 512],
                                   all_steps[i:i + 512])
                      if own_batch_fn is not None else None)
        for j, (y, label) in enumerate(all_labels[i:i + 512]):
            budget = all_budgets[i + j]
            seed_i = all_seed_ids[i + j]
            clf_ok = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
            own_ok = None
            if own_any:
                if own_logits is not None:
                    own_pred = own_logits[j].argmax(dim=-1).item()
                else:
                    own_pred = own_fn(chunk_sub[j], all_seed_rows[i + j],
                                      all_steps[i + j]).argmax(dim=-1).item()
                own_ok = 1.0 if own_pred == label else 0.0
            if judge == 'own':
                main_ok, per_budget_clf_correct[budget] = \
                    own_ok, per_budget_clf_correct[budget] + clf_ok
                per_budget_clf_outcomes[budget].append(clf_ok)
            else:
                main_ok = clf_ok
            per_budget_correct[budget] += main_ok
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(main_ok)
            per_seed_outcomes[seed_i][budget].append(main_ok)
            per_seed_per_class[seed_i][budget][label].append(main_ok)
            if own_any and judge == 'clf':
                per_budget_own_correct[budget] += own_ok
                per_budget_own_outcomes[budget].append(own_ok)

    return _summarize(budget_options, graphs, per_budget_correct,
                      per_budget_total, per_budget_outcomes,
                      per_seed_outcomes, per_seed_per_class,
                      clf_correct=(per_budget_clf_correct
                                   if judge == 'own' else None),
                      clf_outcomes=(per_budget_clf_outcomes
                                    if judge == 'own' else None),
                      own_correct=(per_budget_own_correct
                                   if own_any and judge == 'clf'
                                   else None),
                      own_outcomes=(per_budget_own_outcomes
                                    if own_any and judge == 'clf'
                                    else None))


def _summarize(budget_options, graphs, per_budget_correct, per_budget_total,
               per_budget_outcomes, per_seed_outcomes, per_seed_per_class,
               clf_correct=None, clf_outcomes=None,
               own_correct=None, own_outcomes=None):
    results = {}
    for b in budget_options:
        results[f'acc_{int(b * 100)}%'] = per_budget_correct[b] / max(per_budget_total[b], 1)
    results['acc_overall'] = (sum(per_budget_correct.values()) /
                              max(sum(per_budget_total.values()), 1))
    if clf_correct is not None:
        for b in budget_options:
            results[f'acc_clf_{int(b * 100)}%'] = clf_correct[b] / max(per_budget_total[b], 1)
        results['acc_clf_overall'] = (sum(clf_correct.values()) /
                                      max(sum(per_budget_total.values()), 1))
        results['clf_outcomes'] = {b: list(clf_outcomes[b]) for b in budget_options}
    if own_correct is not None:
        for b in budget_options:
            results[f'acc_own_{int(b * 100)}%'] = own_correct[b] / max(per_budget_total[b], 1)
        results['acc_own_overall'] = (sum(own_correct.values()) /
                                      max(sum(per_budget_total.values()), 1))
        results['own_outcomes'] = {b: list(own_outcomes[b]) for b in budget_options}
    results['n_graphs'] = len(graphs)
    results['outcomes'] = {b: list(per_budget_outcomes[b]) for b in budget_options}
    n_seeds = len(per_seed_outcomes)
    results['per_seed_outcomes'] = {
        i: {b: list(per_seed_outcomes[i][b]) for b in budget_options}
        for i in range(n_seeds)
    }
    results['per_seed_per_class'] = {
        i: {b: {lbl: list(v) for lbl, v in per_seed_per_class[i][b].items()}
            for b in budget_options}
        for i in range(len(per_seed_per_class))
    }
    return results


def budget_stats(outcomes):
    """outcomes -> {mean, std, n} leaf statistics of the eval JSON."""
    arr = np.array(outcomes, dtype=float)
    return {'mean': float(arr.mean()) if len(arr) else 0.0,
            'std': float(arr.std()) if len(arr) else 0.0,
            'n': len(arr)}


@torch.no_grad()
def evaluate_classifier_full(classifier, graphs, labels, device,
                             batch_size=256):
    """Frozen-classifier accuracy on the full graphs (upper-bound reference).


    Graphs are batched directly with dgl.batch; the result is written to
    config.full_graph_ref in the eval JSON as the upper-bound dashed line.
    """
    classifier.eval()
    correct = total = 0
    for start in range(0, len(graphs), batch_size):
        batch_g = dgl.batch(graphs[start:start + batch_size]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[start:start + batch_size].to(device)
        logits = classifier(batch_g, nids)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


_SPLIT_KEYS = ('train', 'valid', 'test')


def merge_eval_json(parts, on_conflict=None):
    """Merge multiple build_eval_json outputs into one multi-split JSON.

    Each part is a single-split product: {split: {strategy: {...}}},
    per_class_{split}, optional acc_clf_reference, and config.
    - split/strategy subtrees are deep-merged (later parts win; on_conflict
      is called with a warning);
    - flat acc_clf_reference is renamed acc_clf_reference_{split};
    - config merges per field: dicts union keys (n_graphs/full_graph_ref
      accumulate per split), scalars: later wins.
    """
    out = {}
    for part in parts:
        for key, value in part.items():
            if key in _SPLIT_KEYS or key.startswith('per_class_'):
                bucket = out.setdefault(key, {})
                for strategy, sub in value.items():
                    if strategy in bucket and on_conflict is not None:
                        on_conflict(f'{key}.{strategy}')
                    bucket[strategy] = sub
            elif key == 'acc_clf_reference':
                split = next((s for s in _SPLIT_KEYS if s in part), None)
                if split is not None:
                    out[f'acc_clf_reference_{split}'] = value
            elif key == 'acc_own_reference':
                split = next((s for s in _SPLIT_KEYS if s in part), None)
                if split is not None:
                    out[f'acc_own_reference_{split}'] = value
            elif key == 'config':
                cfg = out.setdefault('config', {})
                for ck, cv in value.items():
                    if isinstance(cv, dict) and isinstance(cfg.get(ck), dict):
                        cfg[ck].update(cv)
                    else:
                        cfg[ck] = cv
            else:
                out[key] = value
    return out


def build_eval_json(results, split: str, key: str, n_seeds: int,
                    budget_options, config: dict) -> dict:
    """Eval results -> legacy-compatible JSON ({split: {key: {s0..sN, mean}}})."""
    per_seed_stats = {
        f's{seed_i}': {f'{b:.2f}': budget_stats(per_seed[b])
                       for b in budget_options}
        for seed_i, per_seed in results['per_seed_outcomes'].items()
    }
    pooled = {b: [] for b in budget_options}
    for seed_i, per_seed in results['per_seed_outcomes'].items():
        for b in budget_options:
            pooled[b] += per_seed[b]
    per_seed_stats['mean'] = {
        f'{b:.2f}': budget_stats(pooled[b]) for b in budget_options
    }
    all_labels = sorted({lbl for i in results['per_seed_per_class']
                         for b in budget_options
                         for lbl in results['per_seed_per_class'][i][b]})
    per_seed_class_stats = {
        f's{seed_i}': {
            str(lbl): {f'{b:.2f}': budget_stats(
                results['per_seed_per_class'][seed_i][b].get(lbl, []))
                for b in budget_options}
            for lbl in all_labels
        }
        for seed_i in range(n_seeds)
    }
    pooled_per_class = {b: defaultdict(list) for b in budget_options}
    for seed_i in range(n_seeds):
        for b in budget_options:
            for lbl in all_labels:
                pooled_per_class[b][lbl] += \
                    results['per_seed_per_class'][seed_i][b].get(lbl, [])
    per_seed_class_stats['mean'] = {
        str(lbl): {f'{b:.2f}': budget_stats(pooled_per_class[b][lbl])
                   for b in budget_options}
        for lbl in all_labels
    }
    out = {
        split: {key: per_seed_stats},
        f'per_class_{split}': {key: per_seed_class_stats},
        'config': config,
    }
    if 'clf_outcomes' in results:
        out['acc_clf_reference'] = {
            f'{b:.2f}': budget_stats(results['clf_outcomes'][b])
            for b in budget_options
        }
    if 'own_outcomes' in results:
        out['acc_own_reference'] = {
            f'{b:.2f}': budget_stats(results['own_outcomes'][b])
            for b in budget_options
        }
    return out
