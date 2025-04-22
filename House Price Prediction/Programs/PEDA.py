import pandas as pd

def PEDA(d):
    results = pd.DataFrame(columns=["Statistics"] + list(d.columns))

    numeric_stats = [
        "MEAN", "MEDIAN", "MODE", "VARIANCE", "STD_DEV", 
        "MIN", "25%", "50%", "75%", "MAX", "RANGE", "SKEWNESS", "KURTOSIS"
    ]
    
    for stat in numeric_stats:
        row = [stat]
        for col in d.columns:
            if pd.api.types.is_numeric_dtype(d[col]):
                if stat == "MEAN":
                    row.append(d[col].mean())
                elif stat == "MEDIAN":
                    row.append(d[col].median())
                elif stat == "MODE":
                    row.append(d[col].mode().iloc[0] if not d[col].mode().empty else "No Need")
                elif stat == "VARIANCE":
                    row.append(d[col].var())
                elif stat == "STD_DEV":
                    row.append(d[col].std())
                elif stat == "MIN":
                    row.append(d[col].min())
                elif stat == "25%":
                    row.append(d[col].quantile(0.25))
                elif stat == "50%":
                    row.append(d[col].median())
                elif stat == "75%":
                    row.append(d[col].quantile(0.75))
                elif stat == "MAX":
                    row.append(d[col].max())
                elif stat == "RANGE":
                    row.append(d[col].max() - d[col].min())
                elif stat == "SKEWNESS":
                    row.append(d[col].skew())
                elif stat == "KURTOSIS":
                    row.append(d[col].kurt())
            elif pd.api.types.is_object_dtype(d[col]) and stat == "MODE":
                row.append(d[col].mode().iloc[0] if not d[col].mode().empty else "No Need")
            else:
                row.append("No Need")
        results.loc[len(results)] = row

    return results

