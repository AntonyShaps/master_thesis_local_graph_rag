import pandas as pd

path = "./entities.parquet"
rels = pd.read_parquet(path)

print(rels.columns)
print(rels[["description"]].head(20))
for i, text in enumerate(rels["description"]):
    print(f"\n--- row {i} ---")
    print(text)
