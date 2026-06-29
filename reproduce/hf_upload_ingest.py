from huggingface_hub import HfApi, login
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Upload ingested data (VDB, Graphs) to Hugging Face")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["6", "11", "19"],
        help="Collection IDs to upload",
    )
    parser.add_argument(
        "--repo-id",
        default="thang3092004/ebr-rag-final-data-ingest",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Skip pipeline_v2/ (intermediate stage data). Only upload query-time artifacts (~310MB vs 5GB).",
    )
    parser.add_argument(
        "--source-root",
        default="longervideos/ablation-workdirs",
        help="Root folder containing ingested workdirs",
    )
    parser.add_argument(
        "--commit-message",
        default="ingested data",
        help="Commit message / dataset version label on Hugging Face",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Include frame images (huge, regenerable). Default: skip frames.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing repo content before upload (replace stale data).",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(f"Please set HF_TOKEN in .env before running.")

    login(token=token)
    api = HfApi()
    
    print(f"Preparing to upload to: https://huggingface.co/datasets/{args.repo_id}")
    if args.clear:
        print("[CLEAR] Deleting existing repo content (replace stale data)...")
        api.delete_repo(repo_id=args.repo_id, repo_type="dataset", missing_ok=True)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    source_root = (repo_root / args.source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    ignore_patterns = [
        "**/_cache/**",
        "**/.git/**",
        *([] if args.keep_frames else ["**/frames/**"]),
        *(["**/pipeline_v2/**"] if args.artifacts_only else []),
    ]
    print(f"Ignore patterns: {ignore_patterns}")

    for col_id in args.collections:
        matches = sorted(source_root.glob(f"{col_id}-*"))
        if not matches:
            print(f"[SKIP] No ingested data folder found for collection {col_id} under {source_root}")
            continue

        for folder in matches:
            print(f"[UPLOAD] {folder.name}...")
            api.upload_folder(
                folder_path=str(folder),
                repo_id=args.repo_id,
                repo_type="dataset",
                path_in_repo=folder.name,
                commit_message=args.commit_message,
                ignore_patterns=ignore_patterns,
            )

    print(f"\n✅ Uploaded all ingested data for collections: {', '.join(args.collections)}")
    print(f"Repo: {args.repo_id}")
    print(f"Commit message: {args.commit_message!r}")

if __name__ == "__main__":
    main()
