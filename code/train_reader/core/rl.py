"""REINFORCE training loop (shared by the evidence reader).

Reward judgment sources (reward_source): frozen classifier (default), the
reader own evidence-head terminal judgment (--reward-source reader), both
(joint correctness), or soft-both (soft product of correct-class
probabilities, ignores --reward transform).
"""

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed
from .common import (compute_reward, compute_soft_both_reward,  # noqa: F401
                     normalize_advantages)


def train_one_epoch(reader, classifier, builder, optimizer,
                    graphs, labels, batch_size: int, n_seeds: int,
                    device, reward_type: str, ln_k: float, budget_options,
                    neg_scale: float,
                    evidence_lambda: float = 0.0,
                    adv_normalize: bool = True,
                    reward_source: str = 'classifier') -> Dict:
    """One REINFORCE epoch on self-sampled rollouts.

    Loss = mean_i(-A_i * sum_t log pi_t(a_t)); A is the batch-normalized
    advantage (negative side scaled by neg_scale). Reward source:
    - classifier: frozen upper-bound classifier on the revealed subgraph
      (default);
    - reader: the reader own head (evidence only; forces the evidence
      forward);
    - both: classifier (primary) and evidence head (gate) must both be
      correct (tanh clamps to non-positive on miss);
    - soft-both: R = p_clf(y_true) * p_reader(y_true) (evidence only;
      ignores reward_type).
    evidence_lambda: when > 0 and the reader has an evidence stream, adds
    lambda*L_ev (stepwise class CE, never in the policy gradient); total loss
    = L_RL + lambda*L_ev.
    """
    if reward_source in ('reader', 'both', 'soft-both') \
            and not hasattr(reader, 'forward_states_evidence'):
        raise ValueError(
            "reward_source='reader'/'both'/'soft-both' require the evidence "
            "reader (own classification head)")
    use_evidence = (evidence_lambda > 0
                    or reward_source in ('reader', 'both', 'soft-both')) \
        and hasattr(reader, 'forward_states_evidence')
    reader.train()
    if classifier is not None:
        classifier.eval()
    total_R = total_trajs = total_correct = total_loss = 0.0
    ev_correct = ev_total = 0
    ev_final_correct = ev_final_total = 0
    order = list(range(len(graphs)))
    random.shuffle(order)
    pbar = tqdm(range(0, len(order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_indices = order[batch_start:batch_start + batch_size]
        all_states, state_labels, per_state_log_probs = [], [], []
        last_revealed = []
        for local_i in batch_indices:
            g_raw = graphs[local_i]
            label = int(labels[local_i].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()
            budget_ratio = random.choice(list(budget_options))
            B = max(1, int(budget_ratio * n_nodes))
            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                all_states.append(state)
                state_labels.append(y)
                per_state_log_probs.append([])
                last_revealed.append(seed_node)
        if use_evidence:
            all_hc = reader.init_evidence(len(all_states), device)
        ev_losses = []

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)

        for _step in range(max_budget):
            for j in range(n_total):
                if active[j] and all_states[j].is_frontier_empty:
                    st = all_states[j]
                    tp_node = teleport_if_stuck(st)
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
                batch_results, active_hc, active_ev = reader.forward_states_evidence(
                    [all_states[j] for j in active_idx],
                    temperature=reader.temperature,
                    hc_list=[all_hc[j] for j in active_idx],
                    last_revealed=[last_revealed[j] for j in active_idx])
                for k, j in enumerate(active_idx):
                    all_hc[j] = active_hc[k]
                    ev_losses.append(F.cross_entropy(
                        active_ev[k].unsqueeze(0), state_labels[j]))
                    ev_correct += int(active_ev[k].argmax().item()
                                      == state_labels[j].item())
                    ev_total += 1
            else:
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx],
                    temperature=reader.temperature)

            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                per_state_log_probs[j].append(dist.log_prob(action_idx))
                chosen = frontier_nodes[action_idx.item()]
                reveal_node(all_states[j], chosen)
                if use_evidence:
                    last_revealed[j] = chosen
                if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        final_states, final_log_probs, final_labels, final_idx = \
            [], [], [], []
        for j in range(n_total):
            if per_state_log_probs[j]:
                final_idx.append(j)
                final_states.append(all_states[j])
                final_log_probs.append(per_state_log_probs[j])
                final_labels.append(state_labels[j])
        if not final_states:
            continue

        reader_term_logits = None
        if use_evidence:
            _, _, term_ev = reader.forward_states_evidence(
                final_states, reader.temperature,
                hc_list=[all_hc[j] for j in final_idx],
                last_revealed=[last_revealed[j] for j in final_idx])
            reader_term_logits = [t.detach() for t in term_ev]
            for k in range(len(final_states)):
                ev_losses.append(F.cross_entropy(
                    term_ev[k].unsqueeze(0), final_labels[k]))
                ev_final_correct += int(term_ev[k].argmax().item()
                                        == final_labels[k].item())
                ev_final_total += 1

        clf_logits = None
        if reward_source != 'reader':
            logits_chunks = []
            with torch.no_grad():
                for c_start in range(0, len(final_states), 512):
                    chunk = final_states[c_start:c_start + 512]
                    sub_graphs = [builder.build(s)[0] for s in chunk]
                    batch_g = dgl.batch(sub_graphs).to(device)
                    nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                                       device=device)
                    logits_chunks.append(classifier(batch_g, nids))
            clf_logits = torch.cat(logits_chunks, dim=0)

        if reward_source == 'reader':
            primary = torch.stack(reader_term_logits, dim=0)
        else:
            primary = clf_logits
        gate = reader_term_logits \
            if reward_source in ('both', 'soft-both') else None

        raw_rewards = []
        for k in range(len(final_states)):
            correct = 1.0 if primary[k].argmax(dim=-1).item() \
                == final_labels[k].item() else 0.0
            if gate is not None \
                    and gate[k].argmax(dim=-1).item() != final_labels[k].item():
                correct = 0.0
            if reward_source == 'soft-both':
                R = compute_soft_both_reward(primary[k], gate[k],
                                             final_labels[k])
            else:
                R = compute_reward(
                    primary[k:k + 1], final_labels[k], reward_type, ln_k,
                    gate_logits=(gate[k] if gate is not None else None))
            raw_rewards.append(R)
            total_R += R
            total_trajs += 1
            total_correct += correct

        advs = normalize_advantages(raw_rewards, neg_scale) \
            if adv_normalize else list(raw_rewards)

        batch_losses = []
        for k in range(len(final_states)):
            sum_log_prob = torch.stack(final_log_probs[k]).sum()
            batch_losses.append(-advs[k] * sum_log_prob)
        batch_loss = torch.stack(batch_losses).mean()
        if ev_losses:
            batch_loss = batch_loss + evidence_lambda * torch.stack(ev_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += batch_loss.item() * len(batch_losses)

        n = max(total_trajs, 1)
        postfix = {
            'loss': f'{total_loss / n:.4f}',
            'avg_R': f'{total_R / n:.3f}',
            'acc': f'{total_correct / n:.3f}',
        }
        if ev_total:
            postfix['ev_acc'] = f'{ev_correct / ev_total:.3f}'
        if ev_final_total:
            postfix['ev_final'] = f'{ev_final_correct / ev_final_total:.3f}'
        pbar.set_postfix(postfix)
    n = max(total_trajs, 1)
    out = {'loss': total_loss / n, 'avg_R': total_R / n, 'acc': total_correct / n}
    if ev_total:
        out['ev_acc'] = ev_correct / ev_total
    if ev_final_total:
        out['ev_acc_final'] = ev_final_correct / ev_final_total
    return out
