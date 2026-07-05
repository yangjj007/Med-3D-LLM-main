python - <<'PY'
import argparse
import os
import shutil

import pandas as pd


BAD_SHA256 = {
    "6b73561a92a138028e08410617032c9f7a5f14627033d59f1cc0e7000f58a617",
    "6727dbdc5758272d4401fe1e1fcf3bf882fe0b9f4cf04f35d628662a209f22b1",
    "284279e79fc27fff4a2a47766e052be9e42a597665555d326c842dcb09768bee",
    "3cf265ac889c41f2ab4c61b3b1de662147e8e2430eedd2c5703c5eec759c3861",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="./train_sdf_dataset/res512_thre0.5",
        help="Dataset directory containing metadata.csv and *_r512.npz files.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--apply",
        default=True,
        action="store_true",
        help="Actually update metadata.csv and move npz files. Without this flag, only prints actions.",
    )
    parser.add_argument(
        "--delete-files",
        action="store_true",
        help="Permanently delete npz files instead of moving them to _removed_bad_samples.",
    )
    args = parser.parse_args()
    args.apply = True
    metadata_path = os.path.join(args.data_root, "metadata.csv")
    backup_metadata_path = os.path.join(args.data_root, "metadata.csv.bak_before_bad_sample_removal")
    quarantine_dir = os.path.join(args.data_root, "_removed_bad_samples")

    df = pd.read_csv(metadata_path)
    if "sha256" not in df.columns:
        raise ValueError(f"{metadata_path} does not contain a sha256 column.")

    bad_mask = df["sha256"].isin(BAD_SHA256)
    bad_rows = df[bad_mask]

    print(f"metadata: {metadata_path}")
    print(f"total rows: {len(df)}")
    print(f"bad rows found: {len(bad_rows)}")

    if bad_rows.empty:
        print("No matching bad samples found in metadata.")
        return

    for sha in bad_rows["sha256"].tolist():
        flat_path = os.path.join(args.data_root, f"{sha}_r{args.resolution}.npz")
        subdir_path = os.path.join(args.data_root, "sparse_sdf", f"{sha}_r{args.resolution}.npz")
        npz_path = flat_path if os.path.exists(flat_path) else subdir_path

        print(f"\nsha256: {sha}")
        print(f"metadata row: remove")

        if os.path.exists(npz_path):
            if args.delete_files:
                print(f"npz: delete {npz_path}")
                if args.apply:
                    os.remove(npz_path)
            else:
                dst_path = os.path.join(quarantine_dir, os.path.basename(npz_path))
                print(f"npz: move {npz_path} -> {dst_path}")
                if args.apply:
                    os.makedirs(quarantine_dir, exist_ok=True)
                    shutil.move(npz_path, dst_path)
        else:
            print(f"npz: not found for resolution {args.resolution}")

    cleaned = df[~bad_mask]
    print(f"\nrows after removal: {len(cleaned)}")

    if args.apply:
        if not os.path.exists(backup_metadata_path):
            shutil.copy2(metadata_path, backup_metadata_path)
            print(f"metadata backup: {backup_metadata_path}")
        cleaned.to_csv(metadata_path, index=False)
        print("metadata.csv updated.")
    else:
        print("\nDry run only. Re-run with --apply to modify files.")


if __name__ == "__main__":
    main()
PY