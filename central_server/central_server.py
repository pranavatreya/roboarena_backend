"""
Central server for RoboArena distributed evaluation.
"""
import contextlib
import datetime
import random
import threading
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from google.cloud import storage
from scipy.special import expit
import requests
import websockets.sync.client

from config import (
    BASE_CREDIT_CORL,
    BASE_CREDIT_DEFAULT,
    INFINITE_CREDIT_OWNER,
    UCB_TIMES_THRESHOLD,
    WEEKLY_INC_CORL,
    WEEKLY_INC_DEFAULT,
)
from database.connection import initialize_database_connection
from database.schema import (
    EpisodeModel,
    PolicyModel,
    SessionModel,
    UserModel,
)
from logger import logger

# --------------------------------------------------------------------------- #
# Globals / constants
# --------------------------------------------------------------------------- #
SERVER_VERSION = "1.3"
SESSION_TIMEOUT_HOURS = 0.5

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

# Leaderboard algorithm hyper-params
EXCLUDE = {"PI0", "PI0_FAST"}
HYBRID_NUM_T_BUCKETS = 100
EM_ITERS = 60
NUM_RANDOM_SEEDS = 100
SCALE = 200
SHIFT = 1500

# --- Cached A/B evaluations list (for UI page) ---
AB_EVALS_CACHE = {"timestamp": None, "evaluations": []}
AB_EVALS_LOCK = threading.Lock()
AB_EVALS_TTL_SECS = 60  # refresh every ~1 minute

# Leaderboard cache
LEADERBOARD_CACHE = {"timestamp": None, "board": []}
LEADERBOARD_LOCK = threading.Lock()
CACHE_TTL_SECS = 3600

POLICY_ANALYSIS_PATH = "/home/pranavatreya/roboarena_central_server_minimal/output/policy_analysis.json"

rng = np.random.default_rng(0)

# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
SessionLocal = None  # initialised in __main__


# --------------------------------------------------------------------------- #
# Utility: GCS client
# --------------------------------------------------------------------------- #
def get_gcs_client():
    return storage.Client()


# --------------------------------------------------------------------------- #
# Utility: websocket liveness
# --------------------------------------------------------------------------- #
def _ws_policy_alive(ip: str, port: int, timeout: float = 3.0) -> bool:
    try:
        conn = websockets.sync.client.connect(
            f"ws://{ip}:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
        )
        conn.close()
        return True
    except Exception:
        pass # it could be that the server needs a wss connection, we will try that next

    try:
        conn = websockets.sync.client.connect(
            f"wss://{ip}:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
        )
        conn.close()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Cleanup stale sessions
# --------------------------------------------------------------------------- #
def cleanup_stale_sessions():
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale = (
            db.query(SessionModel)
            .filter(
                SessionModel.session_completion_timestamp.is_(None),
                SessionModel.session_creation_timestamp < cutoff,
            )
            .all()
        )
        for s in stale:
            s.session_completion_timestamp = datetime.datetime.utcnow()
            s.evaluation_notes = (s.evaluation_notes or "") + "\nTIMED_OUT"
        if stale:
            db.commit()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Credit bookkeeping helpers
# --------------------------------------------------------------------------- #
def _ensure_weekly_credit(user: UserModel) -> None:
    """
    Top-up evaluation credit with at-most-one-week rollover.

    Logic
    -----
    • If < 1 whole week has elapsed → nothing happens.
    • If ≥ 1 week has elapsed:
        – Keep *at most* one week’s worth of unused credit
          (anything beyond `inc` is considered expired).
        – Add exactly one new week’s increment `inc`.
        – Update `last_credit_update` to `now`.
    • Result: user.eval_credit ∈ [0, 2*inc].
    """
    now = datetime.datetime.utcnow()
    weeks_elapsed = int((now - user.last_credit_update).days / 7)
    if weeks_elapsed <= 0:
        return  # no full week has passed

    # weekly increment depends on CoRL participation
    inc = WEEKLY_INC_CORL if user.participating_corl else WEEKLY_INC_DEFAULT

    # Carry over at most one week's increment
    carry_over = min(user.eval_credit, inc)

    # New balance = carry_over (prev week) + one fresh week
    user.eval_credit = carry_over + inc
    user.last_credit_update = now


def _get_or_create_user(db, email: str) -> UserModel:
    user = db.query(UserModel).filter_by(email=email).first()
    if user:
        _ensure_weekly_credit(user)
        return user
    base = BASE_CREDIT_CORL if False else BASE_CREDIT_DEFAULT
    user = UserModel(
        email=email, participating_corl=False, eval_credit=base, last_credit_update=datetime.datetime.utcnow()
    )
    db.add(user)
    db.flush()
    return user


def _deduct_credit(db, email: str, amount: int = 1) -> None:
    if email == INFINITE_CREDIT_OWNER:
        return
    user = _get_or_create_user(db, email)
    user.eval_credit = max(0, user.eval_credit - amount)


def _reward_credit(db, email: str, amount: int = 1) -> None:
    user = _get_or_create_user(db, email)
    user.eval_credit += amount


# --------------------------------------------------------------------------- #
# Version check
# --------------------------------------------------------------------------- #
@app.route("/version_check", methods=["POST"])
def version_check():
    if (request.get_json() or {}).get("client_version") == SERVER_VERSION:
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "error", "message": "Version mismatch"}), 400


# --------------------------------------------------------------------------- #
# Policy-selection helpers
# --------------------------------------------------------------------------- #
def _ucb_weight(times: int) -> float:
    if times >= UCB_TIMES_THRESHOLD:
        return 1.0
    return np.pow(UCB_TIMES_THRESHOLD / (times + 1), 0.8)


def _sample_policy_A(cands):
    weights = [_ucb_weight(p.times_in_ab_eval or 0) for p in cands]
    return random.choices(cands, weights=weights, k=1)[0]
    #max_weight_idx = 0
    #for i, w in enumerate(weights):
    #    if w > weights[max_weight_idx]:
    #        max_weight_idx = i
    #if random.randint(1, 2) == 2 and weights[max_weight_idx] > 1.0:
    #    return cands[max_weight_idx]
    #return random.choices(cands, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# GET /get_policies_to_compare
# ---------------------------------------------------------------------------
@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    cleanup_stale_sessions()

    evaluator_email = (
        request.args.get("evaluator_email") or request.args.get("evaluator_name")
    )
    if not evaluator_email:
        return jsonify({"error": "evaluator_email missing"}), 400

    eval_location = request.args.get("eval_location", "")
    robot_name = request.args.get("robot_name", "DROID")

    db = SessionLocal()
    try:
        # Ensure evaluator exists / credit up-to-date
        _get_or_create_user(db, evaluator_email)

        # ------------------------- alive policy pool -----------------------
        cand = (
            db.query(PolicyModel)
            .filter(PolicyModel.ip_address.isnot(None), PolicyModel.port.isnot(None))
            .all()
        )
        random.shuffle(cand)
        alive = [p for p in cand if _ws_policy_alive(p.ip_address, p.port)]
        if len(alive) < 2:
            return jsonify({"error": "Fewer than two alive policy servers."}), 400

        # ------------------------- pick UCB policy -------------------------
        elig_A = [
            p
            for p in alive
            if p.owner_name == INFINITE_CREDIT_OWNER
            or (_get_or_create_user(db, p.owner_name).eval_credit > 0)
        ]
        if not elig_A:
            return jsonify({"error": "No policy with available credit."}), 400

        ucb_policy = _sample_policy_A(elig_A)
        uniform_peer = random.choice([p for p in alive if p != ucb_policy])

        # ------------------------- decide visible labels -------------------
        if random.random() < 0.5:
            visible_A, visible_B = ucb_policy, uniform_peer
        else:
            visible_A, visible_B = uniform_peer, ucb_policy

        # ------------------------- create session --------------------------
        sess_uuid = uuid.uuid4()
        internal_note = f"UCB_POLICY={ucb_policy.unique_policy_name}"

        db.add(
            SessionModel(
                session_uuid=sess_uuid,
                evaluation_type="A/B",
                evaluation_location=eval_location,
                evaluator_name=evaluator_email,
                robot_name=robot_name,
                policyA_name=visible_A.unique_policy_name,
                policyB_name=visible_B.unique_policy_name,
                evaluation_notes=internal_note,  # preserves UCB hint
            )
        )
        db.commit()

        response = {
            "session_id": str(sess_uuid),
            "evaluation_type": "A/B",
            "policies": [
                {"label": "A", "ip": visible_A.ip_address, "port": visible_A.port},
                {"label": "B", "ip": visible_B.ip_address, "port": visible_B.port},
            ],
        }
        return jsonify(response), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# POST /upload_eval_data
# --------------------------------------------------------------------------- #
@app.route("/upload_eval_data", methods=["POST"])
def upload_eval_data():
    if not request.form:
        return jsonify({"error": "multipart form-data required"}), 400

    sess_id = request.form.get("session_id")
    if not sess_id:
        return jsonify({"error": "Missing session_id"}), 400

    policy_letter_raw = request.form.get("policy_letter", "")
    letter = policy_letter_raw.split(";", 1)[0].strip().upper()

    db = SessionLocal()
    try:
        sess = db.query(SessionModel).filter_by(session_uuid=sess_id).first()
        if not sess:
            return jsonify({"error": f"No session {sess_id}"}), 400

        # resolve policy name
        policy_name = (
            sess.policyA_name if letter == "A"
            else sess.policyB_name if letter == "B"
            else f"UNLABELED_{letter}"
        )

        # GCS uploads
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        def _upload_if_present(key: str, ext: str):
            f = request.files.get(key)
            if not f:
                return None
            gcs_path = (
                f"{BUCKET_PREFIX}/{sess_id}/{policy_name}_"
                f"{datetime.datetime.utcnow().isoformat()}_{key}.{ext}"
            )
            bucket.blob(gcs_path).upload_from_file(f)
            return gcs_path

        new_episode = EpisodeModel(
            session_id=sess.id,
            policy_name=policy_name,
            command=request.form.get("command", ""),
            binary_success=int(request.form.get("binary_success") or 0)
            if request.form.get("binary_success")
            else None,
            partial_success=float(request.form.get("partial_success") or 0.0)
            if request.form.get("partial_success")
            else None,
            duration=int(request.form.get("duration") or 0)
            if request.form.get("duration")
            else None,
            gcs_left_cam_path=_upload_if_present("video_left", "mp4"),
            gcs_right_cam_path=_upload_if_present("video_right", "mp4"),
            gcs_wrist_cam_path=_upload_if_present("video_wrist", "mp4"),
            npz_file_path=_upload_if_present("npz_file", "npz"),
            policy_ip=request.form.get("policy_ip"),
            policy_port=(
                int(request.form.get("policy_port"))
                if (request.form.get("policy_port") or "").isdigit()
                else None
            ),
            third_person_camera_type=request.form.get("third_person_camera_type"),
            third_person_camera_id=(
                int(request.form.get("third_person_camera_id"))
                if (request.form.get("third_person_camera_id") or "").isdigit()
                else None
            ),
            feedback=policy_letter_raw,
            timestamp=datetime.datetime.utcnow(),
        )
        db.add(new_episode)
        db.commit()
        return jsonify({"status": "success"}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /terminate_session
# ---------------------------------------------------------------------------
@app.route("/terminate_session", methods=["POST"])
def terminate_session():
    data = request.form if request.form else request.get_json()
    if not data:
        return jsonify({"error": "Missing form/JSON data"}), 400

    sess_id = data.get("session_id")
    new_notes = data.get("evaluation_notes", "")

    db = SessionLocal()
    try:
        sess = db.query(SessionModel).filter_by(session_uuid=sess_id).first()
        if not sess:
            return jsonify({"error": f"No session {sess_id}"}), 404

        # ------------------------------------------------------------------
        # Extract stored UCB_POLICY hint **before** modifying evaluation_notes
        # ------------------------------------------------------------------
        ucb_policy_name = None
        for line in (sess.evaluation_notes or "").splitlines():
            if line.startswith("UCB_POLICY="):
                ucb_policy_name = line.split("=", 1)[1].strip()
                break

        # ------------------------------------------------------------------
        # Append evaluator feedback (don’t overwrite internal hint)
        # ------------------------------------------------------------------
        combined_notes = (sess.evaluation_notes or "") + "\n" + new_notes
        sess.evaluation_notes = combined_notes
        sess.session_completion_timestamp = datetime.datetime.utcnow()

        # ------------------------------------------------------------------
        # If session is valid, adjust counts / credit
        # ------------------------------------------------------------------
        if "VALID_SESSION" in new_notes.upper():
            # increment counts for both visible policies
            for pname in (sess.policyA_name, sess.policyB_name):
                pol = db.query(PolicyModel).filter_by(unique_policy_name=pname).first()
                if pol:
                    pol.times_in_ab_eval = (pol.times_in_ab_eval or 0) + 1
                    pol.last_time_evaluated = datetime.datetime.utcnow()

            # debit credit from **ucb_policy’s** owner
            if ucb_policy_name:
                pol_ucb = (
                    db.query(PolicyModel)
                    .filter_by(unique_policy_name=ucb_policy_name)
                    .first()
                )
                if pol_ucb:
                    _deduct_credit(db, pol_ucb.owner_name, 1)

            # reward evaluator
            _reward_credit(db, sess.evaluator_name, 1)

        db.commit()
        return jsonify({"status": "terminated", "session_id": sess_id}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()
        

# ---------------------------------------------------------------------------
# 🆕 Canonical-university mapping + helper
# ---------------------------------------------------------------------------
UNI_CANONICAL = {
    "UNIVERSITY OF CALIFORNIA BERKELEY": "Berkeley",
    "UCB": "Berkeley",
    "BERKELEY": "Berkeley",
    "STANFORD": "Stanford",
    "U PENN": "UPenn",
    "UNIVERSITY OF PENNSYLVANIA": "UPenn",
    "UPENN": "UPenn",
    "UNIVERSITY OF WASHINGTON": "UW",
    "UNVERSITY OF WASHINGTON": "UW",
    "UNIVERSITY OF WASHGINTON": "UW",
    "UW": "UW",
    "MILA": "UMontreal",
    "UNIVERSITY MONTREAL": "UMontreal",
    "UOF MONTREAL": "UMontreal",
    "YONSEI": "Yonsei",
    "UT AUSTIN": "UT Austin",
    "UNIVERSITY OF TEXAS AT AUSTIN": "UT Austin",
}
TARGET_UNIS = [
    "Berkeley",
    "Stanford",
    "UW",
    "UPenn",
    "UMontreal",
    "Yonsei",
    "UT Austin",
]


def canonicalize_uni(raw: str | None) -> str:
    """Return canonical university name or 'Other'."""
    if not raw:
        return "Other"
    upper = raw.upper().strip()
    for k, canon in UNI_CANONICAL.items():
        if k in upper:
            return canon
    return raw # it might be a new institution not in the mapping


def _build_ab_evaluations_payload() -> list[dict]:
    """
    Compute the processed list of A/B evaluations (newest first).
    This is the heavy path that hits Postgres and assembles the payload.
    """
    db = SessionLocal()
    try:
        sessions = (
            db.query(SessionModel)
            .filter(
                SessionModel.evaluation_type == "A/B",
                SessionModel.session_completion_timestamp.isnot(None),
                SessionModel.evaluation_notes.isnot(None),
            )
            .order_by(SessionModel.session_completion_timestamp.desc())
            .all()
        )

        out = []
        for s in sessions:
            # keep only VALID sessions
            if "VALID_SESSION" not in (s.evaluation_notes or "").upper():
                continue

            # parse evaluation_notes
            pref = None
            feedback = None
            for line in (s.evaluation_notes or "").splitlines():
                t = line.strip()
                u = t.upper()
                if u.startswith("PREFERENCE="):
                    pref = u.split("=", 1)[1].strip()
                elif u.startswith("LONGFORM_FEEDBACK="):
                    feedback = t.split("=", 1)[1].strip()

            # fetch episodes for policies A & B
            episodes = (
                db.query(EpisodeModel)
                .filter(
                    EpisodeModel.session_id == s.id,
                    EpisodeModel.policy_name.in_([s.policyA_name, s.policyB_name]),
                )
                .all()
            )
            ep_map = {ep.policy_name: ep for ep in episodes}
            if s.policyA_name not in ep_map or s.policyB_name not in ep_map:
                continue  # incomplete data

            def _policy_block(policy_name: str) -> dict:
                ep = ep_map[policy_name]

                # choose preferred third-person video
                cam_type = (ep.third_person_camera_type or "").lower()
                if "left" in cam_type and ep.gcs_left_cam_path:
                    third_rel = ep.gcs_left_cam_path
                elif "right" in cam_type and ep.gcs_right_cam_path:
                    third_rel = ep.gcs_right_cam_path
                else:
                    third_rel = ep.gcs_left_cam_path or ep.gcs_right_cam_path

                def _url(rel_path: str | None) -> str | None:
                    if not rel_path:
                        return None
                    return f"https://storage.googleapis.com/{BUCKET_NAME}/{rel_path}"

                return {
                    "name": policy_name,
                    "partial_success": ep.partial_success,
                    "wrist_video_url": _url(ep.gcs_wrist_cam_path),
                    "third_person_video_url": _url(third_rel),
                }

            policyA_block = _policy_block(s.policyA_name)
            policyB_block = _policy_block(s.policyB_name)

            # language instruction (any episode works)
            lang_instr = ep_map[s.policyA_name].command or ep_map[s.policyB_name].command

            out.append(
                {
                    "session_id": str(s.session_uuid),
                    "university": canonicalize_uni(s.evaluation_location),
                    "completion_time": s.session_completion_timestamp.isoformat() + "Z",
                    "evaluator_name": s.evaluator_name,
                    "preference": (pref.upper() if isinstance(pref, str) else None),
                    "longform_feedback": feedback,
                    "language_instruction": lang_instr,
                    "policyA": policyA_block,
                    "policyB": policyB_block,
                }
            )

        return out
    finally:
        db.close()


def _ab_evals_refresh_loop():
    """
    Background refresher: recompute the list every AB_EVALS_TTL_SECS.
    Compute first, then acquire the lock only to swap in results.
    """
    while True:
        try:
            data = _build_ab_evaluations_payload()
            now = datetime.datetime.utcnow()
            with AB_EVALS_LOCK:
                AB_EVALS_CACHE["evaluations"] = data
                AB_EVALS_CACHE["timestamp"] = now
            logger.info(f"A/B evals cache refreshed ({len(data)} sessions).")
        except Exception as e:
            logger.error(f"A/B evals recompute failed: {e}")
        time.sleep(AB_EVALS_TTL_SECS)


@app.route("/api/list_ab_evaluations", methods=["GET"])
def list_ab_evaluations():
    """
    Fast path: serve cached results. On cold start (no cache yet),
    compute once synchronously without holding the lock, then swap in.
    """
    with AB_EVALS_LOCK:
        ts = AB_EVALS_CACHE["timestamp"]
        cached = AB_EVALS_CACHE["evaluations"]

    if ts is None:
        # cold start: compute outside the lock, then swap in
        try:
            fresh = _build_ab_evaluations_payload()
            now = datetime.datetime.utcnow()
            with AB_EVALS_LOCK:
                AB_EVALS_CACHE["evaluations"] = fresh
                AB_EVALS_CACHE["timestamp"] = now
            return jsonify({"evaluations": fresh}), 200
        except Exception as e:
            logger.error(f"list_ab_evaluations cold recompute failed: {e}")
            # fall back to whatever we have
            with AB_EVALS_LOCK:
                cached = AB_EVALS_CACHE["evaluations"]
            return jsonify({"evaluations": cached}), 200

    return jsonify({"evaluations": cached}), 200


# Legacy hybrid leaderboard model retained for reference only.
# It is intentionally disabled for leaderboard serving.
def em_hybrid(df,
              iters: int = EM_ITERS,
              step_clip: float = 1.0,
              l2_psi: float = 1e-2,
              l2_theta: float = 1e-2,
              step_decay: float = 0.99,
              tol: float = 1e-4,
              n_restarts: int = 1,
              use_partials: bool = False,
              sigma_partial: float = 0.3,
              partial_weight: float = 1.0): # 2.0 if you want to give partials more weight
    """
    EM for independent‐solve hybrid BT, with optional partial‐success signals.
    If use_partials=True, df must contain 'i_partial' and 'j_partial' in [0,1].
    """
    # ——— Precompute indices & masks ———
    pols   = pd.unique(pd.concat([df.i, df.j]))
    idmap  = {p: k for k, p in enumerate(pols)}
    P      = len(pols)
    i_idx  = df.i .map(idmap).to_numpy()
    j_idx  = df.j .map(idmap).to_numpy()
    y      = df.y .to_numpy()
    win    = (y == 2)
    loss   = (y == 0)
    tie    = (y == 1)

    if use_partials:
        s_i_par = df["i_partial"].to_numpy()
        s_j_par = df["j_partial"].to_numpy()

    best_ll, best_board = -np.inf, None

    for restart in range(n_restarts):
        rng.bit_generator.advance(restart * 1000)

        # ——— Initialize parameters ———
        θ = rng.normal(0., .1, P)
        τ = rng.normal(0., .1, HYBRID_NUM_T_BUCKETS)
        ψ = np.zeros((P, HYBRID_NUM_T_BUCKETS))
        π = np.full(HYBRID_NUM_T_BUCKETS, 1 / HYBRID_NUM_T_BUCKETS)
        ν = 0.5

        def clip_step(x, g, h, clip_val):
            if abs(h) < 1e-8:
                return x
            return x - np.clip(g/h, -clip_val, clip_val)

        # ——— EM loop ———
        for it in range(iters):
            curr_clip = step_clip * (step_decay ** it)

            # E-step: compute solve probabilities
            z_i     = θ[i_idx][:,None] + ψ[i_idx] - τ
            z_j     = θ[j_idx][:,None] + ψ[j_idx] - τ
            solve_i = expit(z_i)
            solve_j = expit(z_j)

            # A/B likelihoods
            p_win  = solve_i * (1 - solve_j)
            p_loss = (1 - solve_i) * solve_j
            p_tie  = 2 * ν * np.sqrt(p_win * p_loss)
            like_ab = (p_win*win[:,None]
                     + p_loss*loss[:,None]
                     + p_tie*tie[:,None])

            # optional partial‐success likelihood
            if use_partials:
                err_i  = (s_i_par[:,None] - solve_i)**2
                err_j  = (s_j_par[:,None] - solve_j)**2
                like_ps = np.exp(-(err_i + err_j)/(2*sigma_partial**2))**partial_weight
                like    = like_ab * like_ps
            else:
                like = like_ab

            # responsibilities γ[n,t]
            γ = π * np.clip(like, 1e-12, None)
            γ /= γ.sum(axis=1, keepdims=True)

            # M-step: update θ
            θ_prev = θ.copy()
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                g = h = 0.0

                for t in range(HYBRID_NUM_T_BUCKETS):
                    # i-slot
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = γ[mi, t]
                    w, l_, tt = win[mi], loss[mi], tie[mi]
                    g  += ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum()
                    h  -= ((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()

                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2

                    # j-slot
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = γ[mj, t]
                    wj, lj, tj = win[mj], loss[mj], tie[mj]
                    g  += ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h  -= ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum()

                    if use_partials:
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                # L2 on θ
                g -= l2_theta * θ[p]
                h -= l2_theta
                θ[p] = clip_step(θ[p], g, h, curr_clip)

            θ -= θ.mean()

            # M-step: update ψ
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                for t in range(HYBRID_NUM_T_BUCKETS):
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = γ[mi, t]
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = γ[mj, t]

                    # A/B
                    w, l_, tt   = win[mi], loss[mi], tie[mi]
                    wj, lj, tj  = win[mj], loss[mj], tie[mj]
                    g = ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum() \
                      + ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h = -(((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()
                         + ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum())

                    # partials
                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                    # L2 on ψ
                    g += l2_psi * ψ[p, t]
                    h -= l2_psi
                    ψ[p, t] = clip_step(ψ[p, t], g, h, curr_clip)

            ψ -= ψ.mean(axis=1, keepdims=True)

            # M-step: update τ
            for t in range(HYBRID_NUM_T_BUCKETS):
                si_t = solve_i[:, t]
                sj_t = solve_j[:, t]
                g    = (γ[:,t]*(si_t + sj_t - 1.0)).sum()
                h    = - (γ[:,t]*(si_t*(1-si_t) + sj_t*(1-sj_t))).sum()
                τ[t] = clip_step(τ[t], g, h, curr_clip)
            τ -= τ.mean()

            # update π, ν
            π = γ.mean(axis=0); π /= π.sum()
            ν = 0.5 * ((p_tie*γ).sum() / max((p_win*γ).sum(), 1e-9))

            if np.max(np.abs(θ - θ_prev)) < tol:
                break

        # finalize restart
        mixlik = (π * like).sum(axis=1)
        ll_cur = np.sum(np.log(mixlik + 1e-12))
        board  = pd.DataFrame({"policy": pols, "score": θ})\
                     .sort_values("score", ascending=False)\
                     .reset_index(drop=True)
        if ll_cur > best_ll:
            best_ll, best_board = ll_cur, board

    return best_board


def fit_bt_davidson(
    df: pd.DataFrame,
    max_iters: int = 200,
    tol: float = 1e-8,
    hess_ridge: float = 1e-6,
) -> tuple[pd.DataFrame, float]:
    """
    Fit standard Bradley-Terry with Davidson ties.

    Outcome model for policy i vs j:
        p(i > j) = exp(theta_i) / Z
        p(j > i) = exp(theta_j) / Z
        p(tie)   = 2 * nu * exp((theta_i + theta_j)/2) / Z
    where Z is the sum of the three numerators and nu > 0.

    We optimize negative log-likelihood with a Newton method using
    analytic gradient/Hessian, and estimate per-policy standard
    deviations from the inverse observed Hessian.
    """
    pols = pd.unique(pd.concat([df.i, df.j]))
    num_policies = len(pols)
    if num_policies == 0:
        return pd.DataFrame(columns=["policy", "score", "std"]), 0.5
    if num_policies == 1:
        one = pd.DataFrame([{"policy": pols[0], "score": 0.0, "std": 0.0}])
        return one, 0.5

    idmap = {p: k for k, p in enumerate(pols)}
    i_idx = df.i.map(idmap).to_numpy()
    j_idx = df.j.map(idmap).to_numpy()
    y = df.y.to_numpy()

    # Fix one policy ability to 0 for identifiability.
    ref_idx = num_policies - 1
    num_theta_free = num_policies - 1
    phi_idx = num_theta_free  # phi = log(nu)
    num_params = num_theta_free + 1

    params = np.zeros(num_params, dtype=float)
    params[phi_idx] = np.log(0.5)

    def unpack_theta(x: np.ndarray) -> np.ndarray:
        theta = np.zeros(num_policies, dtype=float)
        theta[:num_theta_free] = x[:num_theta_free]
        return theta

    def nll_grad_hess(x: np.ndarray):
        theta = unpack_theta(x)
        nu = np.exp(x[phi_idx])

        nll = 0.0
        grad = np.zeros(num_params, dtype=float)
        hess = np.zeros((num_params, num_params), dtype=float)

        for i, j, outcome in zip(i_idx, j_idx, y):
            ti = theta[i]
            tj = theta[j]

            a = np.exp(ti)
            b = np.exp(tj)
            tie_num = 2.0 * nu * np.exp(0.5 * (ti + tj))
            z = a + b + tie_num

            p_i_win = a / z
            p_j_win = b / z
            p_tie = tie_num / z

            v_i_win = np.zeros(num_params, dtype=float)
            v_j_win = np.zeros(num_params, dtype=float)
            v_tie = np.zeros(num_params, dtype=float)

            if i != ref_idx:
                v_i_win[i] = 1.0
                v_tie[i] += 0.5
            if j != ref_idx:
                v_j_win[j] = 1.0
                v_tie[j] += 0.5
            v_tie[phi_idx] = 1.0

            if outcome == 2:
                p_obs = p_i_win
                v_obs = v_i_win
            elif outcome == 0:
                p_obs = p_j_win
                v_obs = v_j_win
            elif outcome == 1:
                p_obs = p_tie
                v_obs = v_tie
            else:
                raise ValueError(f"Unexpected preference value: {outcome}")

            nll -= np.log(np.clip(p_obs, 1e-12, None))

            v_bar = p_i_win * v_i_win + p_j_win * v_j_win + p_tie * v_tie
            grad -= (v_obs - v_bar)

            second_moment = (
                p_i_win * np.outer(v_i_win, v_i_win)
                + p_j_win * np.outer(v_j_win, v_j_win)
                + p_tie * np.outer(v_tie, v_tie)
            )
            hess += second_moment - np.outer(v_bar, v_bar)

        return nll, grad, hess

    for _ in range(max_iters):
        nll, grad, hess = nll_grad_hess(params)
        grad_inf = float(np.linalg.norm(grad, ord=np.inf))
        if grad_inf < tol:
            break

        hess_reg = hess + hess_ridge * np.eye(num_params)
        try:
            step = np.linalg.solve(hess_reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]

        if not np.all(np.isfinite(step)):
            logger.warning("BT-Davidson Newton step became non-finite; stopping early.")
            break

        step_inf = float(np.linalg.norm(step, ord=np.inf))
        if step_inf < tol:
            break

        directional = float(grad @ step)
        alpha = 1.0
        accepted = False
        while alpha >= 1e-8:
            cand = params - alpha * step
            cand[phi_idx] = float(np.clip(cand[phi_idx], -10.0, 10.0))
            cand_nll, _, _ = nll_grad_hess(cand)
            if cand_nll <= nll - 1e-4 * alpha * directional:
                params = cand
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            logger.warning("BT-Davidson line search failed; stopping early.")
            break

        if alpha * step_inf < tol:
            break

    _, _, final_hess = nll_grad_hess(params)
    hess_reg = final_hess + hess_ridge * np.eye(num_params)
    cov_params = np.linalg.pinv(hess_reg, rcond=1e-12)

    theta = unpack_theta(params)
    theta -= theta.mean()

    cov_theta_ref = np.zeros((num_policies, num_policies), dtype=float)
    cov_theta_ref[:num_theta_free, :num_theta_free] = cov_params[
        :num_theta_free, :num_theta_free
    ]

    center = np.eye(num_policies) - (1.0 / num_policies) * np.ones(
        (num_policies, num_policies)
    )
    cov_theta_centered = center @ cov_theta_ref @ center.T
    theta_std = np.sqrt(np.clip(np.diag(cov_theta_centered), 0.0, None))
    theta_std = np.nan_to_num(theta_std, nan=0.0, posinf=0.0, neginf=0.0)

    board = (
        pd.DataFrame({"policy": pols, "score": theta, "std": theta_std})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    tie_nu = float(np.exp(params[phi_idx]))
    return board, tie_nu


def _recompute_leaderboard() -> list[dict]:
    """
    Build preference dataframe -> fit Bradley-Terry with Davidson ties ->
    return list of {policy, score(Elo), std, open_source}.
    """
    db = SessionLocal()
    try:
        # ---------- build preference dataframe ----------
        pairs = []

        sessions = (
            db.query(SessionModel)
              .filter(
                  SessionModel.evaluation_type == "A/B",
                  SessionModel.session_completion_timestamp.isnot(None),
                  SessionModel.evaluation_notes.isnot(None),
              )
              .all()
        )

        for s in sessions:
            if "VALID_SESSION" not in (s.evaluation_notes or "").upper():
                continue
            A, B = s.policyA_name.strip(), s.policyB_name.strip()
            if A.upper() in EXCLUDE or B.upper() in EXCLUDE:
                continue

            pref = None
            for line in (s.evaluation_notes or "").splitlines():
                t = line.strip().upper()
                if t.startswith("PREFERENCE="):
                    pref = {"A": 2, "B": 0, "TIE": 1}.get(
                        t.split("=", 1)[1], None
                    )
                    break
            if pref is None:
                continue

            pairs.append((A, B, pref))

        pref_df = pd.DataFrame(pairs, columns=["i", "j", "y"])
        if pref_df.empty:
            return []

        # Per-policy A/B eval counts (from filtered pref_df)
        counts_i = pref_df["i"].value_counts()
        counts_j = pref_df["j"].value_counts()
        eval_counts = (counts_i.add(counts_j, fill_value=0)).astype(int).to_dict()

        # Legacy hybrid-EM ranking (task-difficulty/task-offset model) is disabled.
        # The public leaderboard now uses classical Bradley-Terry + Davidson ties.
        bt_board, tie_nu = fit_bt_davidson(pref_df)
        logger.info(
            "BT-Davidson fit complete: "
            f"{len(bt_board)} policies, tie_nu={tie_nu:.4f}"
        )

        open_source_rows = db.query(
            PolicyModel.unique_policy_name, PolicyModel.is_in_use
        ).all()
        open_source_map = {
            name: bool(is_open) for name, is_open in open_source_rows
        }

        # ---------- aggregate, transform, tag ----------
        board = []
        for _, row in bt_board.iterrows():
            pol = row["policy"]
            raw_mean = float(row["score"])
            raw_std = float(row["std"])

            elo_mean = round(raw_mean * SCALE + SHIFT)
            elo_std  = round(raw_std  * SCALE, 1)

            board.append({
                "policy": pol,
                "score":  elo_mean,
                "std":    elo_std,
                "open_source": open_source_map.get(pol, False),
                "num_evals": int(eval_counts.get(pol, 0)),
            })

        board.sort(key=lambda d: d["score"], reverse=True)
        return board

    finally:
        db.close()


def _refresh_loop():
    while True:
        try:
            board = _recompute_leaderboard()
            with LEADERBOARD_LOCK:
                LEADERBOARD_CACHE["board"] = board
                LEADERBOARD_CACHE["timestamp"] = datetime.datetime.utcnow()
            logger.info(f"Leaderboard cache refreshed ({len(board)} policies).")
        except Exception as e:
            logger.error(f"Leaderboard recompute failed: {e}")
        time.sleep(CACHE_TTL_SECS)

@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    with LEADERBOARD_LOCK:
        data = {
            "last_updated": (
                LEADERBOARD_CACHE["timestamp"].isoformat() + "Z"
                if LEADERBOARD_CACHE["timestamp"]
                else None
            ),
            "board": LEADERBOARD_CACHE["board"],
        }
    return jsonify(data), 200

@app.route("/api/policy_analysis.json", methods=["GET"])
def serve_policy_analysis():
    try:
        return send_file(POLICY_ANALYSIS_PATH, mimetype="application/json")
    except Exception as e:
        logger.error(f"Cannot serve policy_analysis.json: {e}")
        return jsonify({"error": "analysis report not available"}), 404

# --------------------------------------------------------------------------- #
# App bootstrap
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    db_url = None # hidden in the public release for security reasons
    SessionLocal = initialize_database_connection(db_url)
    logger.info(f"DB connected → {db_url}")
    
    threading.Thread(target=_ab_evals_refresh_loop, daemon=True).start()
    threading.Thread(target=_refresh_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
