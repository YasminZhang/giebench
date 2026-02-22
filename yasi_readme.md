1. download base images
```
python download_base_images.py
```

2. inference with the target model
```
pip install torch diffusers transformers accelerate

# try one image
python run_flux_edits.py --limit=1 --cache-dir=/mnt/shared/yasmin/MODELS
```

In gie_bench.json, each entry’s object_mask is a 2D list with:
Shape: 224 × 224
Rows: len(object_mask) = 224
Columns: len(object_mask[0]) = 224
Total elements: 50,176 (224 × 224)
