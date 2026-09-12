from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PaulineLi/QuantiPhy",
    repo_type="dataset",
    allow_patterns="*.mp4",
    local_dir=r"X:\A\QP\data\QuantiPhy",
)