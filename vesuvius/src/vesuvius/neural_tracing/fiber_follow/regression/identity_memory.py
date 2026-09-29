"""Persistent identity updates judged against the state BEFORE each observation.

The seed is immutable. Recent observation tokens are carried separately and
never used as an identity anchor. There are no label-conditioned writes.
"""
import torch

from .memory import LearnedMemory


class IdentityMemory(LearnedMemory):
    def initial_state(self, batch, device):
        state = super().initial_state(batch, device)
        state['recent'] = torch.zeros(batch, 8, self.cfg.hidden, device=device)
        return state

    def _observe(self, x, state):
        (state, valid, burn, encoded, anchor, anchor_valid, anchor_position,
         anchor_frame, pos, fr, previous, previous_frame, moved, seen) = self.prepare_sequence(x, state)
        slots, recent, probes = state['slots'], state['recent'], []
        for j in range(valid.shape[1]):
            backprop = torch.is_grad_enabled()
            if j < burn:
                backprop = False
            with torch.set_grad_enabled(backprop):
                span = slice(j, j+1)
                observations, anchors = self.write_inputs(
                    encoded[:, span], pos[:, span], fr[:, span], previous[:, span],
                    previous_frame[:, span], moved[:, span], anchor, anchor_position,
                    anchor_frame, anchor_valid)
                # This prediction cannot use the slots resulting from its own write.
                probe = self.probe(observations, slots[:, None], anchors, anchor_valid)
                admission = probe[:, 0, :1].sigmoid()[:, :, None]
                key, value = self.write_keys(torch.cat((observations, anchors), 2))
                proposed = self.write(slots, key[:, 0], value[:, 0], valid[:, j], anchor_valid)
                slots = slots+admission*(proposed-slots)
                recent = torch.where(valid[:, j, None, None], observations[:, 0], recent)
                probes.append(probe)
        return dict(slots=slots, recent=recent, anchor=anchor, anchor_valid=anchor_valid,
                    anchor_position=anchor_position, anchor_frame=anchor_frame,
                    position=pos[:, -1], frame=fr[:, -1], seen=seen,
                    probe=torch.cat(probes, 1))
