import numpy as np
from collections import deque, Counter
import time


# -------------------------------------------------
# COSINE SIMILARITY
# -------------------------------------------------

def cosine_matrix(protos: np.ndarray, emb: np.ndarray):
    """
    Compute cosine similarity.
    Assumes L2-normalized inputs.
    """
    return protos @ emb


# -------------------------------------------------
# TRACK STATE
# -------------------------------------------------

class TrackState:
    def __init__(self, vote_n=5):
        self.votes = deque(maxlen=vote_n)
        self.last_label = "unknown"
        self.last_name = "Unknown"
        self.last_score = 0.0
        self.last_embed_ts = 0.0
        self.last_top_results = []


# -------------------------------------------------
# OPEN SET MATCHER
# -------------------------------------------------

class OpenSetMatcher:

    def __init__(
        self,
        t_accept=0.50,
        t_keep=0.45,
        t_margin=0.07,
        vote_k=3,
        vote_n=5,
        min_face_px=60,
        **kwargs
    ):
        self.t_accept = t_accept
        self.t_keep = t_keep
        self.t_margin = t_margin
        self.vote_k = vote_k
        self.vote_n = vote_n
        self.min_face_px = min_face_px

        self.track = {}

        self.emp_ids = []
        self.emp_names = {}
        self.protos = None
        self.emb_dim = None

    # -------------------------------------------------
    # GALLERY MANAGEMENT
    # -------------------------------------------------

    def update_gallery(self, emp_ids, proto_matrix, emp_names=None):

        if proto_matrix is None or len(emp_ids) == 0:
            self.emp_ids = []
            self.emp_names = {}
            self.protos = None
            self.emb_dim = None
            return

        proto_matrix = proto_matrix.astype(np.float32)

        if proto_matrix.ndim != 2:
            raise ValueError("Gallery embeddings must be 2D")

        # L2 normalize gallery defensively
        norms = np.linalg.norm(proto_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        proto_matrix = proto_matrix / norms

        self.emp_ids = list(emp_ids)
        self.emp_names = emp_names or {}
        self.protos = proto_matrix
        self.emb_dim = proto_matrix.shape[1]

        print(f"[MATCHER] Gallery updated. {len(self.emp_ids)} identities. "
              f"Embedding dim = {self.emb_dim}")

    # -------------------------------------------------
    # SINGLE MATCH
    # -------------------------------------------------

    def _match_once(self, emb: np.ndarray):

        if self.protos is None or len(self.emp_ids) == 0:
            return "unknown", "Unknown", 0.0, 0.0, []

        if emb.shape[0] != self.emb_dim:
            raise RuntimeError(
                f"Probe dim {emb.shape[0]} != gallery dim {self.emb_dim}"
            )

        # Ensure probe is normalized (defensive)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        sims = cosine_matrix(self.protos, emb)

        n_top = min(3, len(sims))
        idx = np.argpartition(sims, -n_top)[-n_top:]
        sorted_idx = idx[np.argsort(sims[idx])][::-1]

        top_results = [{"label": self.emp_ids[i], "name": self.emp_names.get(self.emp_ids[i], "Unknown"), "score": float(sims[i])} for i in sorted_idx]

        if len(sims) == 1:
            s1 = float(sims[0])
            label = self.emp_ids[0]
            name = self.emp_names.get(label, "Unknown")
            return label, name, s1, s1, top_results

        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1]

        s1 = float(sims[best_idx])
        s2 = float(sims[second_idx])
        margin = s1 - s2

        label = self.emp_ids[best_idx]
        name = self.emp_names.get(label, "Unknown")

        return label, name, s1, margin, top_results

    # -------------------------------------------------
    # DECISION LOGIC
    # -------------------------------------------------

    def decide(self, track_id: int, emb: np.ndarray):

        now = time.time()

        if track_id not in self.track:
            self.track[track_id] = TrackState(vote_n=self.vote_n)

        state = self.track[track_id]

        label, name, score, margin, top_results = self._match_once(emb)

        accepted = False

        # Strong accept
        if (
            label != "unknown"
            and score >= self.t_accept
            and margin >= self.t_margin
        ):
            accepted = True

        # Hysteresis keep
        elif (
            state.last_label == label
            and score >= self.t_keep
        ):
            accepted = True

        if not accepted:
            label = "unknown"
            name = "Unknown"

        # Voting
        state.votes.append((label, name))

        if len(state.votes) == 0:
            final_label = "unknown"
            final_name = "Unknown"
        else:
            counts = Counter([v[0] for v in state.votes])
            top_label, top_votes = counts.most_common(1)[0]

            if top_label != "unknown" and top_votes >= self.vote_k:
                final_label = top_label
                final_name = next(v[1] for v in state.votes if v[0] == top_label)
            else:
                final_label = "unknown"
                final_name = "Unknown"

        state.last_label = final_label
        state.last_name = final_name
        state.last_score = score
        state.last_embed_ts = now
        state.last_top_results = top_results

        return final_label, final_name, score, top_results

    # -------------------------------------------------
    # CACHED RESULT (skip embedding)
    # -------------------------------------------------

    def get_cached(self, track_id: int):
        """Return (label, name, score, top_results) from cache, or None if track not seen."""
        state = self.track.get(track_id)
        if state is None:
            return None
        return state.last_label, getattr(state, "last_name", "Unknown"), state.last_score, getattr(state, "last_top_results", [])

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    def cleanup_tracks(self, active_track_ids):

        active_track_ids = set(active_track_ids)

        stale_ids = [
            tid for tid in list(self.track.keys())
            if tid not in active_track_ids
        ]

        for tid in stale_ids:
            del self.track[tid]
