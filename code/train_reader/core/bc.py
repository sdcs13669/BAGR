"""Behavior cloning loop (evidence reader).

Lockstep batched training over graphs; the loss is the teacher-action NLL
plus a stepwise evidence-head CE weighted by evidence_lambda.
"""

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node


def train_one_epoch(reader, optimizer, graph_cache: Dict, train_trajs: List,
                    batch_size: int, device, temperature: float = 1.0,
                    evidence_lambda: float = 0.0) -> Dict:
    """One BC epoch (cross-graph lockstep batching).

    graph_cache: {graph_idx: (graph, label)}；
    evidence_lambda: when > 0 and the reader has an evidence stream, enables
    L_ev (stepwise class CE); total loss = L_act + lambda*L_ev.
    Returns: {'loss': mean step loss, 'bc': same (legacy log key),
              'act_acc': teacher-action agreement,
              'ev_acc': stepwise evidence-head accuracy,
              'ev_acc_final': terminal accuracy (extra forward after budget)
              (ev keys omitted when there is no evidence stream)}
    """
    use_evidence = evidence_lambda > 0 and hasattr(reader, 'forward_states_evidence')
    reader.train()
    total_loss = 0.0
    total_steps = 0
    ev_correct = ev_total = 0
    ev_final_correct = ev_final_total = 0
    act_correct = act_total = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_trajs = train_trajs[batch_start:batch_start + batch_size]

        # ---- Pass 1: create states ----
        all_states: List[GraphState] = []
        state_labels = []
        reveal_orders = []
        step_counters = []
        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            all_states.append(state)
            state_labels.append(torch.tensor([int(label)], device=device))
            reveal_orders.append(traj.reveal_order)
            step_counters.append(0)

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(len(ro) for ro in reveal_orders)
        if use_evidence:
            all_hc = reader.init_evidence(n_total, device)
            last_revealed = [traj.seed_node for traj in batch_trajs]
        step_losses = []
        ev_losses = []

        # ---- Pass 2: cross-graph lockstep ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty
                          and step_counters[j] < len(reveal_orders[j])]
            if not active_idx:
                break

            if use_evidence:
                batch_results, active_hc, active_ev = reader.forward_states_evidence(
                    [all_states[j] for j in active_idx], temperature,
                    [all_hc[j] for j in active_idx],
                    [last_revealed[j] for j in active_idx])
                for k, j in enumerate(active_idx):
                    all_hc[j] = active_hc[k]
                    ev_losses.append(F.cross_entropy(
                        active_ev[k].unsqueeze(0), state_labels[j]))
                    ev_correct += int(active_ev[k].argmax().item()
                                      == state_labels[j].item())
                    ev_total += 1
            else:
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx], temperature=temperature)

            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                teacher_node = reveal_orders[j][step_counters[j]]
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    active[j] = False
                    continue

                step_losses.append(-torch.log(probs[target_idx] + 1e-8))
                act_correct += int(target_idx == probs.argmax().item())
                act_total += 1
                step_counters[j] += 1
                reveal_node(all_states[j], teacher_node)
                if use_evidence:
                    last_revealed[j] = teacher_node
                if step_counters[j] >= len(reveal_orders[j]):
                    active[j] = False

        if not step_losses:
            continue

        if use_evidence:
            _, _, term_ev = reader.forward_states_evidence(
                all_states, temperature, all_hc, last_revealed)
            for k in range(n_total):
                ev_losses.append(F.cross_entropy(
                    term_ev[k].unsqueeze(0), state_labels[k]))
                ev_final_correct += int(term_ev[k].argmax().item()
                                        == state_labels[k].item())
                ev_final_total += 1

        # ---- Pass 3: L_act + λ·L_ev ----
        batch_loss = torch.stack(step_losses).mean()
        if ev_losses:
            batch_loss = batch_loss + evidence_lambda * torch.stack(ev_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = len(step_losses)
        total_loss += batch_loss.item() * n
        total_steps += n

        avg_step_loss = total_loss / max(total_steps, 1)
        postfix = {
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
            'act_acc': f'{act_correct / max(act_total, 1):.3f}',
        }
        if ev_total:
            postfix['ev_acc'] = f'{ev_correct / ev_total:.3f}'
        if ev_final_total:
            postfix['ev_final'] = f'{ev_final_correct / ev_final_total:.3f}'
        pbar.set_postfix(postfix)

    n = max(total_steps, 1)
    out = {'loss': total_loss / n, 'bc': total_loss / n,
           'act_acc': act_correct / max(act_total, 1)}
    if ev_total:
        out['ev_acc'] = ev_correct / ev_total
    if ev_final_total:
        out['ev_acc_final'] = ev_final_correct / ev_final_total
    return out
