import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(SRC_ROOT) not in sys.path:
    # Required for legacy absolute imports (`import datasets`, `import models`)
    # and for `import diffusion.*` trainer registration.
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Cluster cache permissions guard ----
import os, atexit, platform, subprocess
CACHE_DIR = "/scratch/local/ais2t_cache"
PERMS_OCT = 0o2775      # setgid + rwxrwxr-x
GROUP_GID = 3331        # target group id

def _fix_cache_perms():
    try:
        # Only master rank attempts the fix, and only when explicitly opted in.
        if os.environ.get('RANK', '0') != '0':
            return
        if str(os.environ.get('LHNEF_FIX_CACHE_PERMS', os.environ.get('INFD_FIX_CACHE_PERMS', '0'))).lower() not in ('1', 'true', 'yes', 'on'):
            return
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Best-effort; suppress noisy permission errors on shared cluster.
        with open(os.devnull, 'w') as _devnull:
            subprocess.run(["chmod", format(PERMS_OCT, "o"), CACHE_DIR], check=False, stdout=_devnull, stderr=_devnull)
            subprocess.run(["chgrp", f"{GROUP_GID}", CACHE_DIR], check=False, stdout=_devnull, stderr=_devnull)
            subprocess.run(["chmod", "g+s", CACHE_DIR], check=False, stdout=_devnull, stderr=_devnull)
    except Exception as e:
        print(f"[cache] warn: could not set permissions/group: {e}")
if 'htc-' in platform.uname().node:
    atexit.register(_fix_cache_perms)
# ----------------------------------------

from run import parse_cfg_file
from omegaconf import OmegaConf

# IMPORTANT: import as `trainers`, NOT `src.trainers`. Decorators write into the `trainers`
# module's registry; importing the package-qualified path would create a second module
# (and a second, empty registry), causing spurious "trainer not found" KeyErrors.
import trainers  # noqa: E402


def main():
    import argparse
    import yaml
    from omegaconf import open_dict
    from pathlib import Path
    
    def _parse_value(v):
        # Cast common sweep strings (booleans, None, lists) to native YAML types.
        if isinstance(v, str):
            s = v.strip()
            if s in ('True', 'true', 'TRUE'):
                return True
            if s in ('False', 'false', 'FALSE'):
                return False
            if s in ('None', 'none', 'NULL', 'Null', 'null', '~'):
                return None
            try:
                return yaml.safe_load(s)
            except Exception:
                return v
        return v
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True, help='Path to YAML config')
    p.add_argument('--opt', nargs='*', default=[], help='Override as pairs: key value')
    args, unknown = p.parse_known_args()
    # Diagnostics for sweep CLI parsing (rank 0 only).
    try:
        import os as _os_mod
        if _os_mod.environ.get('RANK', '0') == '0':
            if unknown:
                print('[Args][unknown tokens]', unknown)
            else:
                print('[Args][unknown tokens] <none>')
    except Exception:
        pass
    cfg = parse_cfg_file(args.cfg)

    old_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    _MISSING = object()

    def _path_exists(cfg_obj, key: str) -> bool:
        """Return True if OmegaConf path exists (even if its value is None)."""
        try:
            v = OmegaConf.select(cfg_obj, key, default=_MISSING)
        except Exception:
            return False
        return v is not _MISSING

    def _safe_update(cfg_obj, key: str, value):
        """Update cfg; warn (or error under LHNEF_STRICT_OVERRIDES) if key did not exist."""
        existed = _path_exists(cfg_obj, key)
        OmegaConf.update(cfg_obj, key, value)
        if not existed:
            try:
                if os.environ.get('RANK', '0') == '0':
                    msg = f"[Overrides][warn] key did not exist in base cfg (possible typo): {key}={value!r}"
                    if str(os.environ.get('LHNEF_STRICT_OVERRIDES', os.environ.get('INFD_STRICT_OVERRIDES', '0'))).lower() in ('1', 'true', 'yes', 'on'):
                        raise KeyError(msg)
                    print(msg)
            except Exception:
                pass

    if len(args.opt) % 2 != 0:
        raise ValueError(f"--opt expects pairs, got {len(args.opt)} items: {args.opt}")
    _updated_keys = []
    for i in range(0, len(args.opt), 2):
        k, v = args.opt[i: i + 2]
        v_cast = _parse_value(v)
        _safe_update(cfg, k, v_cast)
        _updated_keys.append(k)

    # Generic sweep-style overrides: `--key value` or `--key=value`.
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith('--'):
            i += 1
            continue
        key = tok[2:]
        if '=' in key:
            key, v_str = key.split('=', 1)
            i += 1
        else:
            if i + 1 >= len(unknown) or unknown[i + 1].startswith('--'):
                raise ValueError(f"Flag {tok} is missing a value")
            v_str = unknown[i + 1]
            i += 2
        v = _parse_value(v_str)
        _safe_update(cfg, key, v)
        _updated_keys.append(key)

    # Also accept bare `key=value` tokens (sweeps that omit leading dashes).
    for tok in unknown:
        if tok.startswith('--'):
            continue
        if '=' in tok:
            key, v_str = tok.split('=', 1)
            v = _parse_value(v_str)
            try:
                _safe_update(cfg, key, v)
                _updated_keys.append(key)
            except Exception:
                pass

    # Coerce any stringified booleans/nulls that slipped through.
    for k in _updated_keys:
        try:
            cur = OmegaConf.select(cfg, k)
        except Exception:
            continue
        if isinstance(cur, str):
            coerced = _parse_value(cur)
            if coerced is not cur:
                OmegaConf.update(cfg, k, coerced)

    # Print applied overrides for auditability (rank 0 only in DDP).
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    if _updated_keys and rank == 0:
        print('[Overrides]')
        for k in sorted(set(_updated_keys)):
            try:
                print(f"  {k}: {OmegaConf.select(cfg, k)}")
            except Exception:
                pass

    # Save-root resolution: mirror src/run.py semantics so sweeps via launch.py behave the same.
    # Supports `save_root: from_stage1_dir` / `from_base_ckpt` so stage-2 runs land next to stage-1.
    try:
        with open_dict(cfg):
            if cfg.get("_env", None) is None:
                cfg._env = OmegaConf.create()
            env = cfg._env

            # exp_name: prefer existing env.exp_name; fall back to cfg file stem.
            exp_name = None
            try:
                exp_name = OmegaConf.select(cfg, "_env.exp_name", default=None)
            except Exception:
                exp_name = None
            if exp_name is None or str(exp_name).strip() == "":
                exp_name = Path(args.cfg).stem
                env.exp_name = str(exp_name)

            exp_group = None
            try:
                exp_group = OmegaConf.select(cfg, "_env.exp_group", default=None)
            except Exception:
                exp_group = None

            # Prefer top-level cfg.save_root; fall back to _env.save_root.
            save_root = None
            try:
                save_root = cfg.get("save_root", None)
            except Exception:
                save_root = None
            if save_root is None:
                try:
                    save_root = OmegaConf.select(cfg, "_env.save_root", default=None)
                except Exception:
                    save_root = None
            save_root = "save" if save_root is None else str(save_root)

            save_root_from_ckpt = False
            sr = str(save_root).strip().lower()
            if sr in ("from_base_ckpt", "from_base_ckpt_dir", "base_ckpt_dir", "ckpt_dir"):
                ckpt_p = None
                try:
                    ckpt_p = OmegaConf.select(cfg, "model.args.base_ckpt", default=None)
                except Exception:
                    ckpt_p = None
                if ckpt_p is None:
                    raise ValueError("save_root requests base_ckpt dir but cfg.model.args.base_ckpt is missing.")
                if str(ckpt_p).startswith("/ABS/"):
                    raise ValueError(
                        "cfg.save_root=from_base_ckpt requires a real cfg.model.args.base_ckpt, "
                        f"but it is still the placeholder: {ckpt_p}. "
                        "Pass: --opt model.args.base_ckpt /path/to/best-model.pth"
                    )
                save_root = os.path.dirname(str(ckpt_p))
                save_root_from_ckpt = True
            elif sr in ("from_stage1_dir", "from_stage1", "stage1_dir", "stage1"):
                stage1_dir = None
                try:
                    stage1_dir = OmegaConf.select(cfg, "stage1_dir", default=None)
                except Exception:
                    stage1_dir = None
                if stage1_dir is None:
                    # Fallback: derive from an explicit stage-1 ckpt path.
                    stage1_ckpt = OmegaConf.select(cfg, "sample_vis.stage1_ckpt", default=None)
                    if stage1_ckpt is None:
                        stage1_ckpt = OmegaConf.select(cfg, "sample.stage1_ckpt", default=None)
                    if stage1_ckpt is None:
                        raise ValueError("save_root=from_stage1_dir requires cfg.stage1_dir or sample(_vis).stage1_ckpt")
                    stage1_dir = os.path.dirname(str(stage1_ckpt))
                if str(stage1_dir).startswith("/ABS/"):
                    raise ValueError(
                        "cfg.save_root=from_stage1_dir requires a real cfg.stage1_dir, "
                        f"but it is still the placeholder: {stage1_dir}. "
                        "Pass: --stage1_dir /path/to/STAGE1_RUN_DIR (or override stage1_dir)"
                    )
                save_root = str(stage1_dir)
                save_root_from_ckpt = True

            env.save_root = str(save_root)
            if save_root_from_ckpt:
                # Save directly next to checkpoint/stage1 folder (skip exp_group nesting).
                env.exp_group = ""
            elif exp_group is not None:
                env.exp_group = str(exp_group)

            if getattr(env, "exp_group", None):
                env.save_dir = os.path.join(str(save_root), str(env.exp_group), str(env.exp_name))
            else:
                env.save_dir = os.path.join(str(save_root), str(env.exp_name))
    except Exception as e:
        # Don't hard-fail: better to run with whatever save_dir already exists.
        try:
            if os.environ.get("RANK", "0") == "0":
                print(f"[save_root][warn] could not resolve save_root semantics: {e}")
        except Exception:
            pass

    OmegaConf.set_struct(cfg, old_struct)

    # Keep decoder.z_channels and z_shape[0] in sync with encoder.z_channels.
    def _get(path, default=None):
        try:
            return OmegaConf.select(cfg, path)
        except Exception:
            return default

    zc = _get('model.args.encoder.args.z_channels')
    if zc is not None:
        try:
            zc_int = int(zc)
            # Only sync when a decoder is actually configured. If decoder is null
            # (e.g., pyramid-field renderer), writing decoder.args.* would re-create
            # a dict missing `name` and later crash in models.make().
            dec_spec = _get('model.args.decoder', default=None)
            if dec_spec is not None and (not isinstance(dec_spec, (str, int, float, bool))):
                try:
                    has_name = OmegaConf.select(cfg, 'model.args.decoder.name') is not None
                except Exception:
                    has_name = False
                if has_name:
                    OmegaConf.update(cfg, 'model.args.decoder.args.z_channels', zc_int)
            zshape = list(_get('model.args.z_shape', []) or [])
            if len(zshape) >= 1 and zshape[0] != zc_int:
                zshape[0] = zc_int
                OmegaConf.update(cfg, 'model.args.z_shape', zshape)
        except Exception:
            pass

    # `src/trainers/__init__.py` and `src/datasets/__init__.py` import the optional
    # diffusion package via broad try/except, which silently swallows errors.
    # For sweeps, explicitly import diffusion.train so the failure surfaces and
    # hip_token_* trainers / hip_token_latents datasets get registered.
    if cfg.trainer not in trainers.trainers_dict or "hip" in str(cfg.trainer).lower():
        try:
            import diffusion.train  # noqa: F401 - registers trainers
        except Exception as e:
            raise RuntimeError(
                f"Trainer {cfg.trainer!r} not found. Also failed to import diffusion.train "
                f"(needed for hip_token_* trainers). Underlying error:\n{e}"
            ) from e
    
    # Same rationale for diffusion.data (registers hip_token_latents dataset).
    try:
        import diffusion.data  # noqa: F401
    except Exception as e:
        # Only fatal when the training dataset name actually involves hip latents.
        ds_train_name = None
        try:
            ds_train_name = cfg.get("datasets", {}).get("train", {}).get("name", "")
        except Exception:
            pass
        if ds_train_name and "hip" in str(ds_train_name).lower():
            raise RuntimeError(
                f"Failed to import diffusion.data (needed for {ds_train_name!r} dataset). "
                f"Underlying error:\n{e}"
            ) from e
    if cfg.trainer not in trainers.trainers_dict:
        avail = sorted(list(trainers.trainers_dict.keys()))
        raise KeyError(
            f"Trainer {cfg.trainer!r} not registered after importing diffusion.train. "
            f"Available trainers: {avail}"
        )

    trainer = trainers.trainers_dict[cfg.trainer](cfg)
    trainer.run()


if __name__ == '__main__':
    main()


