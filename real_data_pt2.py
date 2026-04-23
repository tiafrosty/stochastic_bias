import pandas as pd


df_gene_expr = pd.read_csv(
    "GSE76380_series_matrix.txt",
    sep="\t",
    comment="!",   # removes GEO metadata lines
    index_col=0    # genes as index
)


aa = 1


with open("GSE76380_family.soft") as f:
    lines = f.readlines()
    
    
samples = []
current = None

for line in lines:
    line = line.strip()

    if line.startswith("^SAMPLE"):
        if current:
            samples.append(current)
        current = {"GSM": line.split(" = ")[1]}
    
    elif line.startswith("!Sample_title"):
        current["title"] = line.split(" = ")[1]
    
    elif line.startswith("!Sample_characteristics_ch1"):
        val = line.split(" = ")[1]
        if ":" in val:
            k, v = val.split(":", 1)
            current[k.strip()] = v.strip()
if current:
    samples.append(current)

meta_d = pd.DataFrame(samples)

df_gene_expr.columns = meta_d.iloc[:, 1]

aa = 1