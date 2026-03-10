from typing import Dict

import numpy as np


def link_nodules_by_center_distance(tp2nodules: Dict, distance_threshold: float = 30.0) -> Dict:
    """Link nodules across timepoints by nearest-center distance.

    Works backward from the last timepoint, greedily matching each track's
    most-recent nodule to the closest unlinked nodule in the previous timepoint.

    Parameters
    ----------
    tp2nodules : dict
        Mapping of timepoint -> list of (nodule_id, metadata) tuples.
        Metadata must contain a ``"center"`` key with (y, x, z) voxel coords.
        Optionally ``"centers_in_past_exam_ijk_space"`` for registered coords.
    distance_threshold : float
        Maximum voxel distance to consider a match (default 30).

    Returns
    -------
    dict
        Mapping of track_id -> {timepoint: metadata}.
    """
    tracked = {}
    track_id = 0

    rounds = sorted(tp2nodules.keys(), reverse=True)

    # initialise tracks from the last (most recent) timepoint
    for _, metadata in tp2nodules.get(rounds[0], []):
        track_id += 1
        tracked[track_id] = {rounds[0]: metadata}

    # work backward, linking each track to the closest nodule in the prior round
    for i in range(len(rounds) - 1):
        current_round = rounds[i]
        past_round = rounds[i + 1]

        past_nodules = {
            nod_id: meta for nod_id, meta in tp2nodules.get(past_round, [])
        }
        linked_past_ids = set()

        for tid, track_data in list(tracked.items()):
            if current_round not in track_data:
                continue
            current_meta = track_data[current_round]
            # use registered center when available, otherwise fall back to current center
            query_center = current_meta.get("centers_in_past_exam_ijk_space", current_meta["center"])

            best_id, best_dist, best_meta = None, float("inf"), None
            for past_id, past_meta in past_nodules.items():
                if past_id in linked_past_ids:
                    continue
                dist = float(np.linalg.norm(np.array(query_center) - np.array(past_meta["center"])))
                if dist < best_dist:
                    best_dist, best_id, best_meta = dist, past_id, past_meta

            if best_id is not None and best_dist < distance_threshold:
                track_data[past_round] = best_meta
                linked_past_ids.add(best_id)

    # add unlinked screen-detected nodules from earlier rounds as new tracks
    for r in rounds[1:]:
        for nod_id, meta in tp2nodules.get(r, []):
            if not meta.get("screen_detected", False):
                continue
            already_linked = any(
                r in td and td[r]["nodid_in_segmentation"] == meta["nodid_in_segmentation"]
                for td in tracked.values()
            )
            if not already_linked:
                track_id += 1
                tracked[track_id] = {r: meta}

    return tracked
