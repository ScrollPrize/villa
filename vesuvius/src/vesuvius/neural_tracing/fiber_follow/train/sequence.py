"""Episode batches for the sequence follower (models/sequence.py, data.EpisodeSpec).

An episode batch holds every decision of a few simulated traces in step order. Each step's head crop is encoded
(with gradient for the supervised, trailing decisions; without for the steps that only form history), pooled at the
step's committed segment (its head to the next head) into a history token, and the history stream runs causally over
each episode. Every supervised decision is predicted from its own crop, its plane queries and the history tokens of
its episode's earlier steps, offset by their pose in its frame. Returns the decisions' outputs (shared output
contract) and their rows of the batch for the shared losses and metrics.
"""
import torch

from vesuvius.neural_tracing.fiber_follow.models.sequence import relative_pose

ENCODE_CHUNK = 16  # crops per CNN call: one fixed shape for the compiled CNN (train.prepare_training)


def select_rows(batch, rows, count):
    """Row subset of every row-aligned tensor (nested dicts included); other entries are kept as they are."""
    def pick(value):
        if isinstance(value, dict):
            return {key: pick(item) for key, item in value.items()}
        if torch.is_tensor(value) and value.ndim and len(value) == count:
            return value[rows.to(value.device)]
        return value
    return {key: pick(value) for key, value in batch.items()}


def episode_rows(batch):
    """(E, T) batch row of each episode step (-1 past an episode's end), and each row's (episode, step)."""
    index, step = batch['episode_index'].cpu(), batch['episode_step'].cpu()
    episodes = torch.unique(index, sorted=True)
    order = torch.searchsorted(episodes, index)
    table = torch.full((len(episodes), int(step.max())+1), -1, dtype=torch.long)
    table[order, step] = torch.arange(len(index))
    return table, order, step


def encode_rows(model, image, rows):
    """CNN cells of ``image[rows]`` in ``ENCODE_CHUNK``-row calls, the last padded with copies of its first row (instance
    normalization is per crop, so padding never changes a real row)."""
    from vesuvius.neural_tracing.fiber_follow.train.train import fixed_rows
    parts = []
    for start in range(0, len(rows), ENCODE_CHUNK):
        part = rows[start:start+ENCODE_CHUNK]
        parts.append(model.encode(fixed_rows(image[part], ENCODE_CHUNK))[:len(part)])
    return torch.cat(parts)


def committed_local(batch):
    """Each step's committed segment in its own crop frame (N, P, 3) and its mask (N, P)."""
    offset = batch['episode_segment'].double()-batch['crop_pos'][:, None].double()
    return torch.einsum('npi,nij->npj', offset, batch['crop_frame'].double()).float(), batch['episode_segment_mask'].bool()


def episode_forward(model, batch, confidence_threshold, n_commit):
    """(outputs of the supervised decisions, their batch rows)."""
    count = len(batch['hist'])
    device = batch['hist'].device
    table, episode, step = (v.to(device) for v in episode_rows(batch))
    supervised = batch['episode_supervised'].bool()
    rows, others = supervised.nonzero().flatten(), (~supervised).nonzero().flatten()
    image = batch['x']['fine']
    local, valid = committed_local(batch)
    cells = encode_rows(model, image, rows)
    features = torch.zeros(count, 2*cells.shape[1], device=device)
    features = features.index_copy(0, rows, model.step_features(cells, local[rows], valid[rows]).float())
    with torch.no_grad():
        for start in range(0, len(others), ENCODE_CHUNK):
            part = others[start:start+ENCODE_CHUNK]
            features[part] = model.step_features(encode_rows(model, image, part), local[part], valid[part]).float()
    last = valid.sum(1).clamp_min(1)-1
    displacement = local[torch.arange(count, device=device), last]-local[:, 0]
    safe = table.clamp_min(0)
    travelled = batch['travelled'].float()
    states = model.history_states(model.history_input(features[safe], displacement[safe], travelled[safe]))
    past = safe[episode[rows]]
    relative, age = relative_pose(batch['crop_pos'][past], travelled[past], batch['crop_pos'][rows],
                                  batch['crop_frame'][rows], travelled[rows])
    steps = torch.arange(table.shape[1], device=device)
    padding = (steps[None] >= step[rows][:, None]) | (table[episode[rows]] < 0)
    output = model.decide(cells, [s[episode[rows]] for s in states], model.pose(relative, age), padding,
                          confidence_threshold, n_commit)
    return output, select_rows(batch, rows, count)
